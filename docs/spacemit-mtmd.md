> [!IMPORTANT]
> This document describes how to build and run llama.cpp with the SpacemiT SMT backend enabled.

## Build

To build with SMT support, you need:

- the SpacemiT RISC-V toolchain
- `RISCV_ROOT_PATH` pointing to that toolchain
- the SpacemiT ORT package unpacked locally
- `SPACEMIT_ORT_DIR` pointing to the unpacked ORT directory

Download and unpack the required packages:

```bash
wget https://github.com/spacemit-com/toolchain/releases/download/v1.1.2/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz
wget https://github.com/spacemit-com/onnxruntime/releases/download/2.0.2/spacemit-ort.riscv64.2.0.2.tar.gz
```

Then build:

```bash
export RISCV_ROOT_PATH=/path/to/spacemit_toolchain
export SPACEMIT_ORT_DIR=/path/to/spacemit-ort

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_RISCV64_SPACEMIT=ON \
    -DGGML_CPU_REPACK=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF \
    -DGGML_RVV=ON \
    -DGGML_RV_ZVFH=ON \
    -DGGML_RV_ZFH=ON \
    -DGGML_RV_ZICBOP=ON \
    -DGGML_RV_ZIHINTPAUSE=ON \
    -DGGML_RV_ZBA=ON \
    -DCMAKE_INSTALL_PREFIX=build/installed \
    -DCMAKE_TOOLCHAIN_FILE=${PWD}/cmake/riscv64-spacemit-linux-gnu-gcc.cmake \
    -DLLAMA_SERVER_SMT_VISION=ON \
    -DSPACEMIT_ORT_DIR=${SPACEMIT_ORT_DIR}

cmake --build build --parallel "$(nproc)" --config Release

cmake --install build --config Release
```

## Run

After installation, you can start `llama-server` with SMT vision like this:

```bash
./build/installed/bin/llama-server \
  -m multimodal_llm_files/fastvlm-text-0.5B-Q4_1.gguf \
  --vision-backend smt \
  --smt-config-dir ./multimodal_llm_files/ \
  -t 8 \
  --host 0.0.0.0 \
  --port 8080 \
  --warmup
```
