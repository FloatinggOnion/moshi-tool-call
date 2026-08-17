# End-to-End Speech Models with Tool Calling Capabilities

*Author: Jesse-Paul Osemeke*

*Original Paper  (Moshi): [arxiv.org/pdf/2410.00037](https://arxiv.org/pdf/2410.00037)*

## Layout

- `s2s_tools/` -- package source (controller, router, tool registry, transcript buffer, TTS injector, live loop, CLI).
- `docs/` -- architecture diagrams and work log.
- `tests/` -- routing gold set.
- `outputs/` -- generated audio/metrics (not committed).

## Usage

```
uv run s2s --text "What's the weather in Lagos?"
uv run s2s-live --mode wav --wav-in path/to/input.wav
```

The LLM router defaults to a local OpenAI-compatible endpoint (`http://localhost:8000/v1/chat/completions`, e.g. `llama-server`). Set `GROQ_API_URL` and `GROQ_API_KEY` to route through Groq's cloud API instead; `LLM_MODEL` overrides the model name for either.
