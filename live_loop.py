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

from controller import Controller, MinistralRouter
from tools import ToolRegistry
from transcript import RollingTranscriptBuffer

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

    @property
    def is_forcing(self) -> bool:
        with self._lock:
            return bool(self._forced_tokens) or self._forced_hold_steps > 0

    # -- per-frame step -----------------------------------------------------
    def step(self, mic_pcm: np.ndarray) -> tuple[Optional[str], Optional[np.ndarray], bool]:
        """One 80ms frame. Returns (text_piece_or_None, speaker_pcm_or_None,
        was_forcing) -- was_forcing reflects state *before* this step, so
        callers can tell whether the emitted piece was natural or injected."""
        was_forcing = self.is_forcing
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

    def __init__(self) -> None:
        import sounddevice as sd  # local import: optional dependency for --mode wav

        self._sd = sd
        self.input_queue: queue.Queue = queue.Queue()
        self.output_queue: queue.Queue = queue.Queue()

        def on_input(in_data, frames, t, status):
            self.input_queue.put_nowait(in_data[:, 0].astype(np.float32))

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
    ) -> None:
        self.engine = engine
        self.controller = controller
        self.registry = registry
        self.timeline = timeline
        self.source = source
        self.sink = sink

        self._piece_queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._frame_thread: Optional[threading.Thread] = None
        self._turn_active = False
        self._turn_handled = False

    def _frame_loop(self) -> None:
        while not self._stop.is_set():
            mic_pcm = self.source.read()
            if mic_pcm is None:
                self._stop.set()
                break
            piece, out_pcm, was_forcing = self.engine.step(mic_pcm)
            if out_pcm is not None:
                self.sink.write(out_pcm)
            if piece is not None and not was_forcing:
                self._piece_queue.put_nowait(piece)

    async def run(self) -> None:
        buffer = RollingTranscriptBuffer(stability_window_seconds=0.6, min_chunk_chars=6)
        transcript_acc = ""

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
                        transcript_acc += piece
                except queue.Empty:
                    pass

                # Called every tick, not just when new pieces arrived: the
                # stability window is a wall-clock pause detector, so it must
                # keep checking "has this stopped changing yet?" even on ticks
                # with nothing new, or a mid-conversation pause is never seen.
                chunks = buffer.observe(transcript_acc)
                for chunk in chunks:
                    await self._handle_chunk(chunk.text)

                if self._stop.is_set() and self._piece_queue.empty():
                    final_chunks = buffer.flush(final=True)
                    for chunk in final_chunks:
                        await self._handle_chunk(chunk.text)
                    break

                await asyncio.sleep(0.02)
        finally:
            if self._frame_thread is not None:
                self._frame_thread.join(timeout=5)

    async def _handle_chunk(self, text: str) -> None:
        # One reaction per session: without this, the model's tendency to
        # re-ramble near-duplicate text after a forced-injection interruption
        # (see live_loop module docstring) causes the same chunk to be
        # re-routed and re-acted-on indefinitely.
        if not text.strip() or self._turn_active or self._turn_handled:
            return
        self._turn_active = True
        try:
            self.timeline.mark(f"transcript_chunk: {text!r}")
            decision, route_source = await self.controller._router.route(text, self.registry)
            self.timeline.mark(
                f"router_decision: intent={decision.intent} tool_required={decision.tool_required} "
                f"tool={decision.tool} source={route_source}"
            )

            if not decision.tool_required or not decision.tool:
                self.timeline.mark("direct_response: no tool needed, natural generation continues")
                return

            # Only the forced-injection path risks the repeat loop (see
            # module docstring), so only it consumes the one-shot budget --
            # a misheard direct/small-talk turn shouldn't burn it.
            self._turn_handled = True
            self.engine.force_say(HOLDING_PHRASE)
            self.timeline.mark(f"holding_phrase_injected: {HOLDING_PHRASE!r}")

            tool_future = await self.controller._tool_queue.submit(decision.tool, decision.args)
            try:
                tool_result = await tool_future
                self.timeline.mark(f"tool_result_received: {tool_result}")
            except Exception as exc:
                self.timeline.mark(f"tool_error: {exc!r}")
                return

            template = FINAL_ANSWER_TEMPLATES.get(decision.tool)
            final_text = template(tool_result) if template else "Here is what I found."
            self.engine.force_say(final_text)
            self.timeline.mark(f"final_answer_injected: {final_text!r}")

            # Give the injected final-answer tokens time to be consumed by
            # the frame loop before allowing the next turn to start.
            while self.engine.is_forcing:
                await asyncio.sleep(0.05)
            self.timeline.mark("final_answer_playback_complete")
        finally:
            self._turn_active = False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


async def run_wav_mode(args: argparse.Namespace) -> None:
    engine = LiveMoshiEngine(hf_repo=args.hf_repo, quantized=args.quantized)
    registry = ToolRegistry()
    controller = Controller(MinistralRouter(), registry)
    timeline = Timeline()

    source = WavFileSource(Path(args.wav_in), realtime=not args.fast, pad_seconds=args.pad_seconds)
    sink = ArraySink()

    session = LiveSession(engine, controller, registry, timeline, source, sink)
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

    if args.timeline_out:
        Path(args.timeline_out).write_text(
            json.dumps([{"label": label, "t": t} for label, t in timeline.events], indent=2)
        )


async def run_mic_mode(args: argparse.Namespace) -> None:
    engine = LiveMoshiEngine(hf_repo=args.hf_repo, quantized=args.quantized)
    registry = ToolRegistry()
    controller = Controller(MinistralRouter(), registry)
    timeline = Timeline()

    with MicSpeakerIO() as io:
        session = LiveSession(engine, controller, registry, timeline, io, io)
        print("Listening. Press Ctrl+C to stop.")
        try:
            await session.run()
        except KeyboardInterrupt:
            pass

    await controller.shutdown()
    print("timeline:")
    print(timeline.report())


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
    args = parser.parse_args()

    if args.mode == "wav":
        if not args.wav_in:
            parser.error("--wav-in is required for --mode wav")
        asyncio.run(run_wav_mode(args))
    else:
        asyncio.run(run_mic_mode(args))


if __name__ == "__main__":
    main()
