# Speech Synthesis Backends

This directory owns speech output backends for `llama-server` and the
OpenAI-compatible `/v1/audio/speech` endpoint.

The server-facing layer is `tools/server/server-speech.*`. It should expose a
stable request/result boundary and keep HTTP endpoint behavior shared across
backends.

Each backend lives under `tools/speech/backends/<backend-name>/` and should keep
model-specific launchers, converters, kernels, and runtime code inside that
backend directory. Shared speech utilities should be added under `tools/speech`
instead of being copied into every backend.

Current backends:

- `qwen3_tts`: Qwen3-TTS runner, talker driver, codec runtime, ref-bin tooling,
  and K3-specific runtime packaging.
