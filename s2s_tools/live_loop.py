#!/usr/bin/env python3
"""Live full-duplex loop: mic -> Mimi -> Moshi (LmGen) -> transcript tap ->
router -> controller -> (holding phrase -> async tool -> final answer,
force-injected back into Moshi's own text stream) -> Mimi -> speaker.

Two audio-source/sink backends share the same orchestration engine:
  --mode mic   real sounddevice mic/speaker, for interactive use.
  --mode wav   reads a wav file frame-by-frame instead of the mic and writes
               full-duplex output + a timing log, for scripted/offline testing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import rustymimi
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download

from moshi_mlx import models, utils

from .controller import Controller, LLMRouter
from .tools import ToolRegistry
from .transcript import RollingTranscriptBuffer

SAMPLE_RATE = 24000
FRAME = 1920
DEFAULT_HF_REPO = "kyutai/moshika-mlx-q4"
HOLDING_PHRASE = "Let me check that for you."

FINAL_ANSWER_TEMPLATES: dict[str, Callable[[dict], str]] = {
    "get_current_date": lambda r: f"Today's date is {r['date']}.",
    "get_current_time": lambda r: f"It's currently {r['time']}.",
    "get_account_balance": lambda r: (
        f"Your available balance is {r['available_balance']:,.2f} {r['currency']}."
    ),
    "get_weather": lambda r: (
        f"It's {r['current']['temperature_c']} degrees celsius in {r['location']['name']}."
    ),
}


# --------------------------------------------------------------------------
# Timing instrumentation
# --------------------------------------------------------------------------


@dataclass
class Timeline:
    events: list[tuple[str, float]] = field(default_factory=list)

    def mark(self, label: str) -> float:
        t = time.monotonic()
        self.events.append((label, t))
        return t

    def report(self) -> str:
        if not self.events:
            return "(no events)"
        t0 = self.events[0][1]
        return "\n".join(f"  +{t - t0:6.3f}s  {label}" for label, t in self.events)


# --------------------------------------------------------------------------
# Live Moshi engine: one frame in, one frame out, plus forced-phrase injection
# --------------------------------------------------------------------------


class LiveMoshiEngine:
    def __init__(self, hf_repo: str = DEFAULT_HF_REPO, quantized: int = 4, max_steps: int = 4000) -> None:
        if quantized in (4, 8):
            model_file = hf_hub_download(hf_repo, f"model.q{quantized}.safetensors")
        else:
            model_file = hf_hub_download(hf_repo, "model.safetensors")
        tokenizer_file = hf_hub_download(hf_repo, "tokenizer_spm_32k_3.model")
        mimi_file = hf_hub_download(hf_repo, "tokenizer-e351c8d8-checkpoint125.safetensors")

        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)  # type: ignore[arg-type]
        mx.random.seed(299792458)
        self.lm_config = models.config_v0_1()
        self.model = models.Lm(self.lm_config)
        self.model.set_dtype(mx.bfloat16)
        if quantized in (4, 8):
            group_size = 32 if quantized == 4 else 64
            nn.quantize(self.model, bits=quantized, group_size=group_size)
        self.model.load_weights(model_file, strict=True)
        self.model.warmup()

        self.other_codebooks = self.lm_config.other_codebooks
        self.generated_codebooks = self.lm_config.generated_codebooks
        mimi_codebooks = max(self.generated_codebooks, self.other_codebooks)
        self.audio_tokenizer = rustymimi.Tokenizer(mimi_file, num_codebooks=mimi_codebooks)  # type: ignore[arg-type]

        self.gen = models.LmGen(
            model=self.model,
            max_steps=max_steps,
            text_sampler=utils.Sampler(),
            audio_sampler=utils.Sampler(),
            check=False,
            on_text_hook=self._on_text_hook,
        )

        self._lock = threading.Lock()
        self._forced_tokens: list[int] = []
        self._forced_hold_steps = 0
        self._pad_between_tokens = 3
        self._frames_since_force = 0

    # -- forced-phrase injection ------------------------------------------------
    #
    # The dialogue model has no built-in word-pacing protocol (unlike the
    # purpose-built TTS model's new_word/pad state machine), so pacing is
    # approximated: emit one text token, then hold `pad_between_tokens` silent
    # steps before the next. This is a heuristic, not an exact rate match --
    # if the injected phrase sounds too fast/slow, tune pad_between_tokens.
    def _on_text_hook(self, text_token: mx.array) -> None:
        with self._lock:
            if not self._forced_tokens and self._forced_hold_steps <= 0:
                return
            if self._forced_hold_steps > 0:
                self._forced_hold_steps -= 1
                text_token[:] = mx.array([[0]])
                return
            next_id = self._forced_tokens.pop(0)
            text_token[:] = mx.array([[next_id]])
            self._forced_hold_steps = self._pad_between_tokens

    def force_say(self, text: str, pad_between_tokens: int = 3) -> None:
        """Queue `text` to be force-injected as Moshi's own next speech,
        pre-empting whatever it would otherwise say."""
        ids = self.text_tokenizer.encode(text, out_type=int)  # type: ignore[call-arg]
        with self._lock:
            self._forced_tokens = list(ids)
            self._forced_hold_steps = 0
            self._pad_between_tokens = pad_between_tokens
            self._frames_since_force = 0

    def cancel_forcing(self) -> None:
        """Stop mid-injection immediately (e.g. the user barged in) --
        whatever's left in the queue is dropped, natural generation resumes
        on the next step."""
        with self._lock:
            self._forced_tokens = []
            self._forced_hold_steps = 0

    def force_pad(self, num_steps: int) -> None:
        """Force silence for num_steps frames with no queued words -- used as
        a settle period after a forced phrase. The model's own context ending
        in force-injected (not organically sampled) text tends to induce an
        echo/repetition loop once natural generation resumes; a short forced
        silence gives it a clean run-up instead of continuing straight out of
        that context. Heuristic, not a guaranteed fix."""
        with self._lock:
            self._forced_tokens = []
            self._forced_hold_steps = num_steps
            self._frames_since_force = 0

    @property
    def is_forcing(self) -> bool:
        with self._lock:
            return bool(self._forced_tokens) or self._forced_hold_steps > 0

    @property
    def frames_forced(self) -> int:
        """Count of steps spent forcing since the last force_say() call --
        frames_forced * (FRAME/SAMPLE_RATE) is the injected phrase's actual
        audio-time duration, independent of how long it took to compute."""
        with self._lock:
            return self._frames_since_force

    # -- per-frame step -----------------------------------------------------
    def step(self, mic_pcm: np.ndarray) -> tuple[Optional[str], Optional[np.ndarray], bool]:
        """One 80ms frame. Returns (text_piece_or_None, speaker_pcm_or_None,
        was_forcing) -- was_forcing reflects state *before* this step, so
        callers can tell whether the emitted piece was natural or injected."""
        was_forcing = self.is_forcing
        if was_forcing:
            with self._lock:
                self._frames_since_force += 1
        mic_pcm_3d = np.asarray(mic_pcm, dtype=np.float32).reshape(1, 1, -1)
        other_audio_tokens = self.audio_tokenizer.encode_step(mic_pcm_3d)
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[:, :, : self.other_codebooks]
        text_token = self.gen.step(other_audio_tokens[0])
        text_token = text_token[0].item()

        piece = None
        if text_token not in (0, 3):
            piece = self.text_tokenizer.id_to_piece(text_token).replace("▁", " ")  # type: ignore[arg-type]

        out_pcm = None
        audio_tokens = self.gen.last_audio_tokens()
        if audio_tokens is not None and self.generated_codebooks > 0:
            audio_tokens_np = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            out_pcm = self.audio_tokenizer.decode_step(audio_tokens_np)[0, 0]

        return piece, out_pcm, was_forcing


# --------------------------------------------------------------------------
# Audio source/sink backends
# --------------------------------------------------------------------------


class WavFileSource:
    """Feeds a wav file into the engine frame-by-frame, as a stand-in mic."""

    def __init__(self, path: Path, realtime: bool = True, pad_seconds: float = 0.0) -> None:
        pcm, sr = sphn.read(str(path), sample_rate=SAMPLE_RATE)
        assert sr == SAMPLE_RATE
        pcm = np.asarray(pcm)[0].astype(np.float32)
        if pad_seconds > 0:
            # Real conversation gives Moshi room to respond after the question
            # ends; a wav that stops the instant speech does starves it of the
            # frames it needs to start generating a reply.
            pcm = np.concatenate([pcm, np.zeros(int(pad_seconds * SAMPLE_RATE), dtype=np.float32)])
        n_frames = len(pcm) // FRAME
        pcm = pcm[: n_frames * FRAME]
        self._frames = pcm.reshape(n_frames, FRAME)
        self._idx = 0
        self.realtime = realtime
        self._next_due = time.monotonic()

    def read(self) -> Optional[np.ndarray]:
        if self._idx >= len(self._frames):
            return None
        frame = self._frames[self._idx]
        self._idx += 1
        if self.realtime:
            self._next_due += FRAME / SAMPLE_RATE
            delay = self._next_due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return frame


class ArraySink:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def to_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.frames)


class MicSpeakerIO:
    """Real sounddevice mic/speaker backend for interactive use. Not
    exercised by the automated test harness -- requires a live speaker."""

    # engine.step() runs slower than real-time on modest hardware (measured
    # ~9 steps/s vs. the ~12.5 steps/s mic input arrives at), so an unbounded
    # queue here just accumulates an ever-growing backlog of stale audio --
    # the model falls further behind every second and never catches up to
    # what's actually being said "now". Bounding the queue and dropping the
    # oldest frame when full keeps it processing near-live audio instead,
    # at the cost of skipping some input while it's behind.
    MAX_QUEUED_INPUT_FRAMES = 4

    def __init__(self) -> None:
        import sounddevice as sd  # local import: optional dependency for --mode wav

        self._sd = sd
        self.input_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self.MAX_QUEUED_INPUT_FRAMES)
        self.output_queue: queue.Queue = queue.Queue()

        def on_input(in_data, frames, t, status):
            frame = in_data[:, 0].astype(np.float32)
            try:
                self.input_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.input_queue.get_nowait()  # drop the oldest, make room for the newest
                except queue.Empty:
                    pass
                try:
                    self.input_queue.put_nowait(frame)
                except queue.Full:
                    pass

        def on_output(out_data, frames, t, status):
            try:
                out_data[:, 0] = self.output_queue.get(block=False)
            except queue.Empty:
                out_data.fill(0)

        self.in_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME, callback=on_input)
        self.out_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME, callback=on_output)

    def __enter__(self):
        self.in_stream.start()
        self.out_stream.start()
        return self

    def __exit__(self, *exc):
        self.in_stream.stop()
        self.out_stream.stop()

    def read(self) -> Optional[np.ndarray]:
        return self.input_queue.get()

    def write(self, frame: np.ndarray) -> None:
        self.output_queue.put_nowait(frame)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


class LiveSession:
    def __init__(
        self,
        engine: LiveMoshiEngine,
        controller: Controller,
        registry: ToolRegistry,
        timeline: Timeline,
        source,
        sink,
        barge_in_rms_threshold: float = 0.02,
        barge_in_frames: int = 3,
    ) -> None:
        self.engine = engine
        self.controller = controller
        self.registry = registry
        self.timeline = timeline
        self.source = source
        self.sink = sink
        self.barge_in_rms_threshold = barge_in_rms_threshold
        self.barge_in_frames = barge_in_frames

        self._piece_queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._frame_thread: Optional[threading.Thread] = None
        self._turn_active = False
        self._turn_handled = False

        self._interrupt_flag = threading.Event()
        self._barge_in_run = 0
        self._current_turn_task: Optional[asyncio.Task] = None
        self._current_tool_future = None

        # Direct/small-talk turns don't consume the one-shot tool-turn
        # budget (see _maybe_start_turn), so without this a model stuck
        # repeating filler with no real speech to react to floods the
        # router with an identical chunk on every repeat.
        self._last_direct_text: Optional[str] = None
        self._last_direct_time = 0.0
        self._direct_cooldown_seconds = 3.0

        self.buffer = RollingTranscriptBuffer(stability_window_seconds=0.6, min_chunk_chars=6)
        self.transcript_acc = ""
        self.metrics: list[dict] = []

        # For post-hoc splice-discontinuity analysis (wav mode only): index
        # into sink.frames (not the raw step count -- steps during the
        # initial acoustic-delay warmup produce no audio and are never
        # written to the sink, so the two counts diverge) of each
        # was_forcing False->True / True->False transition.
        self._sink_frame_index = 0
        self.forcing_transitions: list[tuple[int, str]] = []

    def _frame_loop(self) -> None:
        was_forcing_prev = False
        while not self._stop.is_set():
            mic_pcm = self.source.read()
            if mic_pcm is None:
                self._stop.set()
                break
            piece, out_pcm, was_forcing = self.engine.step(mic_pcm)
            if out_pcm is not None:
                self.sink.write(out_pcm)
                if was_forcing != was_forcing_prev:
                    self.forcing_transitions.append(
                        (self._sink_frame_index, "forcing_start" if was_forcing else "forcing_end")
                    )
                    was_forcing_prev = was_forcing
                self._sink_frame_index += 1
            if piece is not None and not was_forcing:
                self._piece_queue.put_nowait(piece)

            # Barge-in: user speaking over the holding phrase / final answer.
            # Only armed during an active turn -- ordinary silence-detector
            # noise shouldn't matter outside a turn since there's nothing to
            # interrupt. Requires sustained energy (not a single loud frame)
            # to avoid false triggers on clicks/pops.
            if self._turn_active and was_forcing:
                rms = float(np.sqrt(np.mean(np.asarray(mic_pcm, dtype=np.float32) ** 2)))
                if rms > self.barge_in_rms_threshold:
                    self._barge_in_run += 1
                    if self._barge_in_run >= self.barge_in_frames and not self._interrupt_flag.is_set():
                        self.engine.cancel_forcing()  # stop injected audio immediately, don't wait for the async side
                        self._interrupt_flag.set()
                else:
                    self._barge_in_run = 0
            else:
                self._barge_in_run = 0

    async def run(self) -> None:
        self._frame_thread = threading.Thread(target=self._frame_loop, daemon=True)
        self._frame_thread.start()

        try:
            # An unconditional loop with an explicit break, rather than a
            # `while not stop or not empty` guard: the guard can flip false at
            # the exact moment stop+empty become simultaneously true, exiting
            # before the loop body's final-flush branch ever runs.
            while True:
                try:
                    while True:
                        piece = self._piece_queue.get_nowait()
                        self.transcript_acc += piece
                except queue.Empty:
                    pass

                if self._interrupt_flag.is_set():
                    self._interrupt_flag.clear()
                    await self._handle_interruption()

                # Called every tick, not just when new pieces arrived: the
                # stability window is a wall-clock pause detector, so it must
                # keep checking "has this stopped changing yet?" even on ticks
                # with nothing new, or a mid-conversation pause is never seen.
                chunks = self.buffer.observe(self.transcript_acc)
                for chunk in chunks:
                    self._maybe_start_turn(chunk.text)

                if self._stop.is_set() and self._piece_queue.empty():
                    final_chunks = self.buffer.flush(final=True)
                    for chunk in final_chunks:
                        self._maybe_start_turn(chunk.text)
                    if self._current_turn_task is not None and not self._current_turn_task.done():
                        await self._current_turn_task
                    break

                await asyncio.sleep(0.02)
        finally:
            if self._frame_thread is not None:
                self._frame_thread.join(timeout=5)

    def _maybe_start_turn(self, text: str) -> None:
        # One reaction per session: without this, the model's tendency to
        # re-ramble near-duplicate text after a forced-injection interruption
        # (see live_loop module docstring) causes the same chunk to be
        # re-routed and re-acted-on indefinitely.
        if not text.strip() or self._turn_active or self._turn_handled:
            return
        # Same guard for the direct/small-talk path, which isn't covered by
        # _turn_handled: if the model repeats the same filler with nothing
        # new to react to, don't re-route on every repeat.
        now = time.monotonic()
        if text == self._last_direct_text and (now - self._last_direct_time) < self._direct_cooldown_seconds:
            return
        self._turn_active = True
        self._current_turn_task = asyncio.create_task(self._handle_chunk(text))

    async def _handle_interruption(self) -> None:
        """User spoke over the holding phrase or final answer: cancel the
        in-flight turn (tool call included) and clear conversation state so
        the new speech gets a fresh re-plan, not a stale re-route."""
        self._log("interrupted_by_user", "[!]      interrupted -- re-planning")

        if self._current_tool_future is not None and not self._current_tool_future.done():
            self._current_tool_future.cancel()
        if self._current_turn_task is not None and not self._current_turn_task.done():
            self._current_turn_task.cancel()
            try:
                await self._current_turn_task
            except asyncio.CancelledError:
                pass

        self.engine.cancel_forcing()  # idempotent; frame loop already did this, but harmless to repeat
        self.buffer.reset()
        self.transcript_acc = ""
        self._turn_active = False
        self._turn_handled = False
        self._current_turn_task = None
        self._current_tool_future = None
        self._log("re_plan_ready", "[ready]  listening again")

    async def _watch_forcing_complete(self) -> int:
        while self.engine.is_forcing:
            await asyncio.sleep(0.05)
        return self.engine.frames_forced

    def _log(self, label: str, display: str) -> float:
        """Mark the timeline and print live to the console -- the timeline
        report only prints at session end, so this is the only place to
        follow the conversation while --mode mic is actually running."""
        t = self.timeline.mark(label)
        print(display, flush=True)
        return t

    async def _handle_chunk(self, text: str) -> None:
        turn_metrics: dict = {"user_text": text}
        try:
            t_chunk = self._log(f"transcript_chunk: {text!r}", f"[heard]  {text}")
            decision, route_source = await self.controller._router.route(text, self.registry)
            t_decision = self._log(
                f"router_decision: intent={decision.intent} tool_required={decision.tool_required} "
                f"tool={decision.tool} source={route_source}",
                f"[router] intent={decision.intent} tool={decision.tool}",
            )
            turn_metrics["router_latency_s"] = t_decision - t_chunk
            turn_metrics["intent"] = decision.intent
            turn_metrics["tool"] = decision.tool

            if not decision.tool_required or not decision.tool:
                self._log("direct_response: no tool needed, natural generation continues", "[direct] no tool needed")
                self._last_direct_text = text
                self._last_direct_time = time.monotonic()
                return

            # Only the forced-injection path risks the repeat loop (see
            # module docstring), so only it consumes the one-shot budget --
            # a misheard direct/small-talk turn shouldn't burn it.
            self._turn_handled = True
            self.engine.force_say(HOLDING_PHRASE)
            t_holding = self._log(f"holding_phrase_injected: {HOLDING_PHRASE!r}", f"[moshi]  {HOLDING_PHRASE}")
            turn_metrics["latency_to_holding_phrase_s"] = t_holding - t_chunk

            # Run the tool call and the holding-phrase playback watcher
            # concurrently: the tool may finish before or after the phrase
            # has actually finished playing. Either way, the final answer
            # must not cut the holding phrase off mid-word (force_say()
            # unconditionally replaces whatever's currently queued), so both
            # are awaited together before the final answer is spoken.
            self._current_tool_future = await self.controller._tool_queue.submit(decision.tool, decision.args)
            holding_watch = asyncio.ensure_future(self._watch_forcing_complete())
            try:
                tool_result, holding_frames = await asyncio.gather(self._current_tool_future, holding_watch)
            except asyncio.CancelledError:
                holding_watch.cancel()
                raise
            except Exception as exc:
                holding_watch.cancel()
                t_err = self._log(f"tool_error: {exc!r}", f"[tool]   error: {exc}")
                turn_metrics["tool_error"] = repr(exc)
                turn_metrics["tool_latency_s"] = t_err - t_holding
                return
            t_tool = self._log(f"tool_result_received: {tool_result}", f"[tool]   {tool_result}")
            turn_metrics["tool_latency_s"] = t_tool - t_holding
            turn_metrics["holding_phrase_audio_s"] = holding_frames * FRAME / SAMPLE_RATE

            template = FINAL_ANSWER_TEMPLATES.get(decision.tool)
            final_text = template(tool_result) if template else "Here is what I found."
            self.engine.force_say(final_text)
            t_final = self._log(f"final_answer_injected: {final_text!r}", f"[moshi]  {final_text}")
            turn_metrics["latency_to_final_answer_s"] = t_final - t_chunk
            turn_metrics["final_answer_text"] = final_text

            final_frames = await self._watch_forcing_complete()
            self.timeline.mark("final_answer_playback_complete")
            final_audio_s = final_frames * FRAME / SAMPLE_RATE
            word_count = max(1, len(final_text.split()))
            turn_metrics["final_answer_audio_s"] = final_audio_s
            turn_metrics["final_answer_pace_s_per_word"] = final_audio_s / word_count
            # Natural conversational speech runs roughly 0.3-0.5s/word; pace
            # far outside that band is a proxy for "will sound unnatural"
            # (too clipped/rushed if low, too draggy/robotic if high).
            turn_metrics["final_answer_pace_natural"] = 0.2 <= turn_metrics["final_answer_pace_s_per_word"] <= 0.7

            # Settle period: forced injection leaves the model's context
            # ending in text it never organically sampled, which tends to
            # induce an echo/repetition loop once natural generation resumes
            # (see force_pad docstring). A brief forced silence breaks that.
            self.engine.force_pad(12)
            await self._watch_forcing_complete()
        except asyncio.CancelledError:
            turn_metrics["cancelled"] = True
            raise
        finally:
            self._turn_active = False
            self._current_tool_future = None
            self.metrics.append(turn_metrics)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def compute_splice_discontinuities(
    frames: list[np.ndarray], transitions: list[tuple[int, str]], window: int = 3
) -> list[dict]:
    """RMS-amplitude jump across each natural<->forced audio splice point --
    a large jump is a proxy for an audible, unnatural cut. `frames` is the
    session's full sequence of decoded 1920-sample output frames; a `None`
    frame (no audio produced that step) is skipped when building the window."""
    results = []
    for frame_index, kind in transitions:
        before = [f for f in frames[max(0, frame_index - window) : frame_index] if f is not None]
        after = [f for f in frames[frame_index : frame_index + window] if f is not None]
        if not before or not after:
            continue
        rms_before = float(np.sqrt(np.mean(np.concatenate(before) ** 2)))
        rms_after = float(np.sqrt(np.mean(np.concatenate(after) ** 2)))
        results.append(
            {
                "frame_index": frame_index,
                "kind": kind,
                "rms_before": rms_before,
                "rms_after": rms_after,
                "rms_jump": abs(rms_after - rms_before),
            }
        )
    return results


def print_metrics_summary(session: "LiveSession") -> None:
    if not session.metrics:
        print("metrics: no turns handled")
        return
    print("metrics:")
    for m in session.metrics:
        print(f"  turn: {m.get('user_text', '')!r}")
        for key in (
            "router_latency_s",
            "latency_to_holding_phrase_s",
            "tool_latency_s",
            "latency_to_final_answer_s",
            "holding_phrase_audio_s",
            "final_answer_audio_s",
            "final_answer_pace_s_per_word",
            "final_answer_pace_natural",
        ):
            if key in m:
                print(f"    {key}: {m[key]}")


def append_metrics(metrics_out: str, session: "LiveSession", splices: list[dict], run_label: str) -> None:
    path = Path(metrics_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for m in session.metrics:
            record = {"run": run_label, **m}
            f.write(json.dumps(record) + "\n")
        if splices:
            f.write(json.dumps({"run": run_label, "splice_discontinuities": splices}) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


async def run_wav_mode(args: argparse.Namespace) -> None:
    engine = LiveMoshiEngine(hf_repo=args.hf_repo, quantized=args.quantized)
    registry = ToolRegistry()
    controller = Controller(LLMRouter(), registry)
    timeline = Timeline()

    source = WavFileSource(Path(args.wav_in), realtime=not args.fast, pad_seconds=args.pad_seconds)
    sink = ArraySink()

    session = LiveSession(
        engine, controller, registry, timeline, source, sink,
        barge_in_rms_threshold=args.barge_in_threshold,
        barge_in_frames=args.barge_in_frames,
    )
    timeline.mark("session_start")
    await session.run()
    timeline.mark("session_end")

    await controller.shutdown()

    out_pcm = sink.to_array()
    out_path = Path(args.wav_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(out_path), out_pcm, SAMPLE_RATE)

    print(f"wrote {out_path} ({len(out_pcm) / SAMPLE_RATE:.2f}s)")
    print("timeline:")
    print(timeline.report())

    splices = compute_splice_discontinuities(sink.frames, session.forcing_transitions)
    if splices:
        print("splice discontinuities:")
        for s in splices:
            print(f"  frame {s['frame_index']} ({s['kind']}): rms_jump={s['rms_jump']:.4f}")

    print_metrics_summary(session)

    if args.timeline_out:
        Path(args.timeline_out).write_text(
            json.dumps([{"label": label, "t": t} for label, t in timeline.events], indent=2)
        )

    if args.metrics_out:
        append_metrics(args.metrics_out, session, splices, run_label=Path(args.wav_in).stem)


async def run_mic_mode(args: argparse.Namespace) -> None:
    engine = LiveMoshiEngine(hf_repo=args.hf_repo, quantized=args.quantized)
    registry = ToolRegistry()
    controller = Controller(LLMRouter(), registry)
    timeline = Timeline()

    with MicSpeakerIO() as io:
        session = LiveSession(
            engine, controller, registry, timeline, io, io,
            barge_in_rms_threshold=args.barge_in_threshold,
            barge_in_frames=args.barge_in_frames,
        )
        print("Listening. Press Ctrl+C to stop.")
        try:
            await session.run()
        except (KeyboardInterrupt, asyncio.CancelledError):
            # asyncio.run()'s SIGINT handling cancels the running task, which
            # raises CancelledError here (not KeyboardInterrupt) -- either
            # way, Ctrl+C means stop and still print the summary below.
            pass

    await controller.shutdown()
    print("timeline:")
    print(timeline.report())
    print_metrics_summary(session)

    if args.metrics_out:
        append_metrics(args.metrics_out, session, [], run_label="mic_session")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live full-duplex Moshi + router/controller loop.")
    parser.add_argument("--mode", choices=["mic", "wav"], default="wav")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--quantized", type=int, default=4, choices=[4, 8])
    parser.add_argument("--wav-in", help="Input wav for --mode wav (stand-in mic audio).")
    parser.add_argument("--wav-out", default="outputs/live_loop_out.wav")
    parser.add_argument("--timeline-out", help="Optional path to dump timeline events as JSON.")
    parser.add_argument("--fast", action="store_true", help="Don't pace --mode wav playback in real time.")
    parser.add_argument(
        "--pad-seconds",
        type=float,
        default=12.0,
        help="Trailing silence appended after --wav-in so Moshi has room to respond and the tool round-trip can complete.",
    )
    parser.add_argument(
        "--barge-in-threshold",
        type=float,
        default=0.02,
        help="Mic RMS energy above which a frame counts as user speech during barge-in detection.",
    )
    parser.add_argument(
        "--barge-in-frames",
        type=int,
        default=3,
        help="Consecutive above-threshold frames required to confirm a barge-in (debounces clicks/pops).",
    )
    parser.add_argument(
        "--metrics-out",
        default="outputs/metrics.jsonl",
        help="Append per-turn latency/naturalness metrics as JSON lines here. Empty string to disable.",
    )
    args = parser.parse_args()

    try:
        if args.mode == "wav":
            if not args.wav_in:
                parser.error("--wav-in is required for --mode wav")
            asyncio.run(run_wav_mode(args))
        else:
            asyncio.run(run_mic_mode(args))
    except KeyboardInterrupt:
        # asyncio.run() re-raises the original KeyboardInterrupt here even
        # after run_mic_mode already handled the cancellation and printed
        # its summary -- this just stops it from also printing a traceback.
        pass


if __name__ == "__main__":
    main()
