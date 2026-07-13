if (WIN32)
    message(FATAL_ERROR "Qwen3-TTS requires POSIX process and socket APIs")
endif()

if (NOT DEFINED SPACEMIT_ORT_DIR)
    message(FATAL_ERROR "SPACEMIT_ORT_DIR must be set for Qwen3-TTS")
endif()

set(QWEN3_TTS_ORT_INCLUDE_DIR "${SPACEMIT_ORT_DIR}/include")
set(QWEN3_TTS_ORT_LIB_DIR "${SPACEMIT_ORT_DIR}/lib")

if (EXISTS "${QWEN3_TTS_ORT_LIB_DIR}/libonnxruntime.so")
    set(QWEN3_TTS_ORT_LIBRARY "${QWEN3_TTS_ORT_LIB_DIR}/libonnxruntime.so")
elseif (EXISTS "${QWEN3_TTS_ORT_LIB_DIR}/libonnxruntime.a")
    set(QWEN3_TTS_ORT_LIBRARY "${QWEN3_TTS_ORT_LIB_DIR}/libonnxruntime.a")
else()
    message(FATAL_ERROR "Could not find libonnxruntime.{so,a} in SPACEMIT_ORT_DIR='${SPACEMIT_ORT_DIR}'")
endif()

add_library(qwen3-tts-runtime STATIC
    qwen3_tts_runtime.cpp
    qwen3_tts_runtime.h
    qwen3_tts_gguf.h
    qwen3_tts_protocol.h
)
target_include_directories(qwen3-tts-runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(qwen3-tts-runtime SYSTEM PRIVATE
    ${QWEN3_TTS_ORT_INCLUDE_DIR}
)
target_link_libraries(qwen3-tts-runtime PRIVATE
    llama
    ggml-base
    ${QWEN3_TTS_ORT_LIBRARY}
    ${CMAKE_DL_LIBS}
    Threads::Threads
)
target_compile_features(qwen3-tts-runtime PRIVATE cxx_std_17)

add_executable(llama-qwen3-tts-runner qwen3_tts_runner.cpp)
target_link_libraries(llama-qwen3-tts-runner PRIVATE qwen3-tts-runtime llama)
target_compile_features(llama-qwen3-tts-runner PRIVATE cxx_std_17)

add_executable(llama-qwen3-tts-talker
    qwen3_tts_talker.cpp
    qwen3_tts_gguf.h
    qwen3_tts_protocol.h
)
target_include_directories(llama-qwen3-tts-talker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu
)
target_link_libraries(llama-qwen3-tts-talker PRIVATE llama ggml-cpu ggml-base Threads::Threads)
target_compile_features(llama-qwen3-tts-talker PRIVATE cxx_std_17)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv" AND GGML_CPU_RISCV64_SPACEMIT)
    target_compile_options(llama-qwen3-tts-talker PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-march=rv64gcv_zfh_zvfh_zba_zicbop_zihintpause>
        $<$<CXX_COMPILER_ID:GNU>:-mabi=lp64d>
    )
endif()

if (TARGET llama-server)
    add_dependencies(llama-server llama-qwen3-tts-runner llama-qwen3-tts-talker)
endif()
