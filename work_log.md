# Moshi Ministral Work Log

#### 06/07/2026

- Run first local test `uv run python -m moshi_mlx.local_web -q 4 --hf-repo "kyutai/moshika-mlx-q4"`
- Time to first audio: 0s - 2.23s
- Latency 3s - 5s, increases as conversation grows longer
- No thermal throttling observed

#### 07/07/2026

- Run local model (Qwen-3.5-0.8B) (~82 tokens/sec)
- Tested on 20 gold set questions in `routing_tests.json`. All pass.

#### 11/07/2026

- Build tool registry + async tool queue (`tools.py`) with mock date/time, dummy account balance, and live weather (Open-Meteo).
- Build deterministic controller state machine and split planning/execution paths (`controller.py`).
- Add rolling transcript buffer + chunk stability logic (`transcript.py`) for safe snapshotting before routing.
- Add typed-input and transcript dry-run modes in `main.py` (`--text`, `--transcript-dry-run`).
- Dry-run transcript snapshots and verify expected Ministral planning behavior without wiring full Moshi loop.

#### 17/07/2026

- Build `moshi_speech.py` (`MoshiSpeechInjector`) to render assistant text as speech via the offline DSM TTS model (text-prefixed generation), for the holding phrase and post-tool final answer.
- Fix real bugs found while getting the first render to run: weight loading needed `load_pytorch_weights` (the raw checkpoint uses a pth-style depformer layout, not the mlx-native one `load_weights` expects), and output PCM needed conversion from `mx.array` to `numpy` before `sphn.write_wav`.
- Validate both `--render-holding-phrase` ("Let me check that for you.") and `--render-final-answer` (canned balance answer) end-to-end in `main.py`; both produce well-formed wav output.

#### 17/08/2026

- Swap the router model from Ministral to a locally-served Qwen-3.5-0.8B (`llama-server`, OpenAI-compatible). Found and fixed two real routing bugs surfaced by the small reasoning model: the raw `/v1/completions` endpoint let `<think>` draft JSON leak and get parsed before the real answer, and the JSON extractor's "last object wins" fix initially also matched nested `args` objects as their own candidates. Switched to `/v1/chat/completions` with `enable_thinking: false` and a two-shot (tool vs. small-talk) example; added a `tool_required`/`response_mode` consistency invariant in `RoutingDecision.from_dict` since the small model sets them independently. Gold set now at 14/20 (remaining 6 failures are tools that were never implemented, e.g. `convert_currency`, `set_reminder`, plus one small-talk misroute) -- all 4 real tools (date/time/weather/balance) route correctly.
- Build `live_loop.py`: a full-duplex engine (`LiveMoshiEngine`, `kyutai/moshika-mlx-q4`) wiring mic -> Mimi -> `LmGen.step()` -> transcript tap on Moshi's own inner-monologue text -> router -> controller -> holding phrase -> async tool -> final answer -> Mimi -> speaker. Two backends share one orchestration engine: `--mode wav` (deterministic file-driven harness, used for all testing here) and `--mode mic` (real sounddevice mic/speaker, untested interactively -- no way to drive a live microphone from this session).
- Holding-phrase/final-answer injection is live-forced directly into Moshi's own text stream via `LmGen`'s `on_text_hook` (teacher-forcing, the same mechanism the offline TTS model uses internally, adapted here for the first time to the live dialogue model, which has no built-in word-pacing protocol -- paced with a `pad_between_tokens` heuristic instead).
- Fixed a real orchestration bug: the transcript-buffer polling loop's exit condition could go false at the exact moment stop+empty became simultaneously true, skipping the final-flush branch entirely and silently dropping the whole transcript. Also fixed the stability-window check to run every poll tick (not just when new text arrived), and added a one-shot-per-session guard on the *tool-triggering* path only (the model tends to re-ramble near-duplicate text after a forced-injection interruption, which was re-triggering the same tool call indefinitely).
- Generated synthetic "user question" audio with the already-validated offline TTS renderer (no live mic available in this session) and fed it through the real live loop for all 3 scripted tool intents:
  - **Balance**: full success -- correct tool call, correct final answer synthesized from live tool output, holding phrase injected 1ms before the tool result arrived.
  - **Time**: full success -- same, holding phrase injected 2ms before the tool result.
  - **Weather**: routing and holding-phrase timing worked correctly, but the tool call itself failed because Moshi's own transcription misheard "Abuja" as "Boujab" (an unresolvable place name) -- a real STT-accuracy limitation of the synthetic test voice on the quantized model, not a pipeline bug.
- Checkpoint: **holding phrase reliably starts before the tool result returns**, confirmed by timestamped events across every trigger (12 repeats in an early weather run, plus the balance and time runs). Whether the resumed speech "feels natural" could not be judged directly (no way to listen to audio in this session) -- the user should listen to `outputs/live_balance_out.wav` / `outputs/live_time_out.wav` / `outputs/live_weather_out.wav` directly.
- Found a real hardware constraint (consistent with the very first 06/07 test): this machine runs the live loop at roughly 3-5x slower than real-time (12.5 steps/sec target), so genuine interactive mic/speaker use would likely feel laggy. Per-turn output audio pacing itself is unaffected (governed by `pad_between_tokens`, not compute speed) -- only true live back-and-forth would show the lag. Recommend the semi-scripted fallback (this same pipeline, offline-rendered audio) for a live audience demo unless the throughput gap is addressed first.
