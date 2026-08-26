set(TARGET llama-media)

add_library(${TARGET} STATIC
    media-worker.cpp
    media-worker.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_SERVER_SMT_MTMD)
    if(WIN32)
        message(FATAL_ERROR "LLAMA_SERVER_SMT_MTMD is not supported on Windows because the SMT audio backend depends on POSIX APIs")
    endif()

    if(NOT DEFINED SPACEMIT_ORT_DIR)
        message(FATAL_ERROR "SPACEMIT_ORT_DIR must be set when LLAMA_SERVER_SMT_MTMD=ON")
    endif()

    set(SPACEMIT_ORT_INC_DIR "${SPACEMIT_ORT_DIR}/include")
    set(SPACEMIT_ORT_LIB_DIR "${SPACEMIT_ORT_DIR}/lib")
    set(SPACEMIT_ORT_SAMPLE_DIR "${SPACEMIT_ORT_DIR}/samples")

    if(NOT EXISTS "${SPACEMIT_ORT_SAMPLE_DIR}/spine_vision_engine.cc")
        message(FATAL_ERROR "Could not find spine_vision_engine.cc in SPACEMIT_ORT_DIR='${SPACEMIT_ORT_DIR}'")
    endif()
    if(NOT EXISTS "${SPACEMIT_ORT_SAMPLE_DIR}/spine_llm_argparser.cc")
        message(FATAL_ERROR "Could not find spine_llm_argparser.cc in SPACEMIT_ORT_DIR='${SPACEMIT_ORT_DIR}'")
    endif()

    if(EXISTS "${SPACEMIT_ORT_LIB_DIR}/libonnxruntime.so")
        set(ONNXRUNTIME_LIB "${SPACEMIT_ORT_LIB_DIR}/libonnxruntime.so")
    elseif(EXISTS "${SPACEMIT_ORT_LIB_DIR}/libonnxruntime.a")
        set(ONNXRUNTIME_LIB "${SPACEMIT_ORT_LIB_DIR}/libonnxruntime.a")
    else()
        message(FATAL_ERROR "Could not find libonnxruntime.{so,a} in SPACEMIT_ORT_DIR='${SPACEMIT_ORT_DIR}'")
    endif()

    target_sources(${TARGET} PRIVATE
        smt/smt-audio-wrapper.cpp
        smt/smt-audio-wrapper.h
        smt/multi-asr/multi-asr-common.cpp
        smt/multi-asr/multi-asr-common.h
        smt/multi-asr/multi-asr-encoder.cpp
        smt/multi-asr/multi-asr-encoder.h
        smt/multi-asr/multi-asr-decoder.cpp
        smt/multi-asr/multi-asr-decoder.h
        smt/multi-asr/multi-asr-orchestrator.cpp
        smt/multi-asr/multi-asr-orchestrator.h
        smt/multi-asr/multi-asr-service.cpp
        smt/multi-asr/multi-asr-service.h
        smt/smt-media-common.cpp
        smt/smt-media-common.h
        smt/smt-profile.h
        smt/smt-tts-wrapper.cpp
        smt/smt-tts-wrapper.h
        smt/smt-vision-preprocess.cpp
        smt/smt-vision-preprocess.h
        smt/smt-vision-wrapper.cpp
        smt/smt-vision-wrapper.h
        smt/qwen3-tts/qwen3_tts_gguf.h
        smt/qwen3-tts/qwen3_tts_runtime.cpp
        smt/qwen3-tts/qwen3_tts_runtime.h
        smt/qwen3-tts/qwen3_tts_talker.cpp
        smt/qwen3-tts/qwen3_tts_talker.h
        ${SPACEMIT_ORT_SAMPLE_DIR}/spine_vision_engine.cc
        ${SPACEMIT_ORT_SAMPLE_DIR}/spine_llm_argparser.cc
    )
    target_include_directories(${TARGET} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/smt
        ${CMAKE_CURRENT_SOURCE_DIR}/smt/multi-asr
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/common
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tools/mtmd
        ${CMAKE_SOURCE_DIR}/vendor
        ${CMAKE_SOURCE_DIR}/ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/smt/qwen3-tts
        ${SPACEMIT_ORT_SAMPLE_DIR}
        ${SPACEMIT_ORT_INC_DIR}
    )
    target_link_libraries(${TARGET} PRIVATE mtmd llama ggml-cpu ${ONNXRUNTIME_LIB})
    if(CMAKE_DL_LIBS)
        target_link_libraries(${TARGET} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv" AND GGML_CPU_RISCV64_SPACEMIT)
        get_target_property(QWEN3_TTS_RISCV_OPTIONS ggml-cpu COMPILE_OPTIONS)
        if(QWEN3_TTS_RISCV_OPTIONS)
            list(FILTER QWEN3_TTS_RISCV_OPTIONS INCLUDE REGEX "^-m(arch|abi)=")
            set_source_files_properties(
                smt/qwen3-tts/qwen3_tts_talker.cpp
                PROPERTIES COMPILE_OPTIONS "${QWEN3_TTS_RISCV_OPTIONS}"
            )
        endif()
    endif()
endif()
