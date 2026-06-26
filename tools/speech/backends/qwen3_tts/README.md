# SpaceMIT Speech Qwen3-TTS Backend

This directory builds the SpaceMIT Qwen3-TTS speech backend inside `llama.cpp`.
It contains the Qwen3-TTS runtime, the q3tts runner, the talker driver, and the
prompt reference converter.

The public HTTP entrypoint is `llama-server` with the OpenAI-compatible
`/v1/audio/speech` endpoint. The TTS model zoo component should call that
endpoint; it should not embed this runtime code.

## Build

Qwen3-TTS depends on ONNX Runtime for codec decoding and optionally copies the
SpaceMIT EP shared library into the runtime prefix. Pass dependency roots
explicitly when building from source packages:

```bash
cmake -B build-q3tts -S . \
  -DLLAMA_BUILD_SPEECH=ON \
  -DQ3TTS_ONNXRUNTIME_ROOT=/path/to/onnxruntime/install \
  -DQ3TTS_SPACEMIT_EP_ROOT=/path/to/spacemit-ep/install \
  -DGGML_CPU_RISCV64_SPACEMIT=ON

cmake --build build-q3tts --parallel 8 --target q3tts-install
```

`q3tts-install` installs `llama-server`, `q3tts-ref-to-bin`, `q3tts-run`,
`q3tts-runner`, `talker_driver.headmain`, `talker_driver.qwen3tts-k3`, and the
shared runtime libraries needed by the launcher into `CMAKE_INSTALL_PREFIX`.
`q3tts-run` is an internal/diagnostic runner used by the server-side speech
backend.

`LLAMA_BUILD_Q3TTS=ON` is kept as a compatibility alias for existing build
scripts, but new integrations should use `LLAMA_BUILD_SPEECH=ON`.

## Run

```bash
llama-server \
  -m ${HOME}/.cache/models/tts/qwen3-tts/gguf/qwen3-tts-0.6b-talker-qkv-gateup-q8_0-side.gguf \
  --media-backend smt \
  --smt-config-dir ${HOME}/.cache/models/tts/qwen3-tts \
  --host 127.0.0.1 \
  --port 8090 \
  --alias qwen3-tts \
  --tts-speaker-file ${HOME}/.cache/models/tts/qwen3-tts/refs/default.spk.bin \
  -t 4 \
  -c 128
```

The server exposes an OpenAI-compatible `/v1/audio/speech` endpoint for the
model-zoo TTS backend. `--tts-speaker-file` accepts a `.spk.bin` reference bin
or a `.wav` reference audio file. If it is omitted, the backend looks for a
default reference under `${smt-config-dir}/refs`.

The runtime defaults to the Q8 side-preserving talker GGUF because it is the
validated quality default. Q4 remains available as an explicit performance mode:

```bash
Q3TTS_TALKER_GGUF=qwen3-tts-0.6b-talker-qkv-gateup-q4_0-side.gguf llama-server ...
```
