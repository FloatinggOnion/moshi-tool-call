# Work Log

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
- Dry-run transcript snapshots and verify expected routing behavior without wiring full Moshi loop.

#### 17/07/2026

- Build `moshi_speech.py` (`MoshiSpeechInjector`) to render assistant text as speech via the offline DSM TTS model, for the holding phrase and post-tool final answer.
- Fix weight loading (needs `load_pytorch_weights`, not `load_weights`) and PCM-to-numpy conversion before `sphn.write_wav`.
- Validate `--render-holding-phrase` and `--render-final-answer` end-to-end in `main.py`.

#### 17/08/2026

- Swap router model from Ministral to a locally-served Qwen-3.5-0.8B (`llama-server`, OpenAI-compatible). Fixed two routing bugs: `<think>` draft JSON leaking into the parsed output, and JSON extraction matching nested `args` objects. Switched to `/v1/chat/completions` with `enable_thinking: false` and a two-shot example; added a `tool_required`/`response_mode` consistency invariant. Gold set 14/20 (remaining failures are unimplemented tools); all 4 real tools route correctly.
- Build `live_loop.py`: full-duplex engine (`LiveMoshiEngine`, `kyutai/moshika-mlx-q4`) wiring mic -> Mimi -> `LmGen.step()` -> transcript tap on Moshi's inner-monologue -> router -> controller -> holding phrase -> async tool -> final answer -> Mimi -> speaker. `--mode wav` (file-driven test harness) and `--mode mic` (sounddevice, untested interactively).
- Holding-phrase/final-answer injection uses `LmGen`'s `on_text_hook` (teacher-forcing), adapted from the offline TTS model's mechanism to the live dialogue model; paced with a `pad_between_tokens` heuristic since the dialogue model has no built-in word-pacing protocol.
- Fixed a transcript-buffer final-flush race (loop could exit before the final-flush branch ran) and added a one-shot-per-session guard on the tool-triggering path (the model re-rambles near-duplicate text after a forced-injection interruption, causing repeat tool calls).
- Tested all 3 scripted tool intents (synthetic TTS-rendered audio as mic input): balance and time succeeded end-to-end (correct tool call, correct final answer); weather's routing/timing worked but the tool failed because Moshi misheard "Abuja" as "Boujab" (STT accuracy, not a pipeline bug).
- Checkpoint confirmed: holding phrase reliably starts before the tool result returns, across every trigger.
- Hardware runs the live loop ~3-5x slower than real-time; per-turn audio pacing itself is unaffected (governed by `pad_between_tokens`), only true live back-and-forth would show it. Recommend the semi-scripted fallback for a live demo unless addressed.
- **Interruption handling.** Added mic-energy barge-in detection: sustained RMS above threshold while forced audio is playing cancels injection immediately and triggers a full re-plan (cancels the in-flight tool call, resets transcript state). Turn handling restructured into a cancellable `asyncio.Task` to support this. Validated against the real engine and detection code directly; not validated against a true overlapping-speech wav file.
- **Moshi/router concurrency.** Confirmed contention: both processes share the same Apple M2 GPU. Benchmarked three configs -- GPU-default (1.41x Moshi slowdown, router latency ~0.5-2s), CPU-only unlimited threads (1.54x, worse -- shifts contention onto shared CPU cores), CPU-only `--threads 2` (1.12x, but router latency jumps to 1-12s). Kept GPU-default as the better overall tradeoff. Added `LLMRouter._request_gate`, a shared semaphore serializing all router calls regardless of local hardware split. Recommendation: a cloud-hosted router removes the contention entirely and is the real fix if both need to be fast at once.
- **Latency/naturalness metrics.** Every tool-triggering turn records latency to holding-phrase start, tool latency, latency to final answer, actual audio-time durations (frame-count based), and a naturalness proxy (`final_answer_pace_s_per_word`, flagged against a 0.2-0.7s/word band) plus RMS-jump splice discontinuity at natural/forced audio boundaries. Appended to `outputs/metrics.jsonl`. Surfaced and fixed a real bug: `force_say()` was cutting the holding phrase off before it played when the tool responded fast (i.e. always, for local tools) -- now waits for holding-phrase playback to finish before speaking the final answer. Balance-intent rerun: holding phrase got its full 2.16s; final-answer pace came out to 0.96s/word (outside natural band -- numeric-heavy text tokenizes into many pieces under the fixed `pad_between_tokens` heuristic), splice discontinuities small (rms_jump 0.0003-0.0017). Next tuning step: lower or adapt `pad_between_tokens` to phrase length.

#### 17/08/2026 (cont'd)

- Reorganized the codebase: source into `s2s_tools/` package, docs into `docs/`, `routing_tests.json` into `tests/`. Added `pyproject.toml` packaging (hatchling) with `s2s`/`s2s-live` console-script entry points.
- Renamed `MinistralRouter` -> `LLMRouter` (not tied to any specific model). Removed `MINISTRAL_API_URL`/`MINISTRAL_MODEL`. Router now defaults to the local endpoint unless `GROQ_API_URL` is set, in which case `GROQ_API_KEY` is required and used as a bearer token against Groq's cloud API. `LLM_MODEL` overrides the model name for either modality.
