include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-tools.cpp
    server-tools.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_BUILD_SPEECH)
    target_sources(${TARGET} PRIVATE
        server-speech-backend.h
        server-speech.cpp
        server-speech.h
        server-speech-qwen3-tts.cpp
        server-speech-qwen3-tts.h
    )
endif()

if(LLAMA_SERVER_SMT_VISION)
    if(WIN32)
        message(FATAL_ERROR "LLAMA_SERVER_SMT_VISION is not supported on Windows because the SMT audio backend depends on POSIX dlopen APIs")
    endif()

    if(NOT DEFINED SPACEMIT_ORT_DIR)
        message(FATAL_ERROR "SPACEMIT_ORT_DIR must be set when LLAMA_SERVER_SMT_VISION=ON")
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

    target_compile_definitions(${TARGET} PRIVATE LLAMA_SERVER_SMT_VISION=1)
    target_sources(${TARGET} PRIVATE
        server-smt-vision.cpp
        server-smt-vision.h
        ../mtmd/smt-audio-wrapper.cpp
        ../mtmd/smt-vision-preprocess.cpp
        ../mtmd/smt-vision-wrapper.cpp
        ${SPACEMIT_ORT_SAMPLE_DIR}/spine_vision_engine.cc
        ${SPACEMIT_ORT_SAMPLE_DIR}/spine_llm_argparser.cc
    )
    set(ONNXRUNTIME_LIB "")
    if(EXISTS "${SPACEMIT_ORT_LIB_DIR}/libonnxruntime.so")
        set(ONNXRUNTIME_LIB "${SPACEMIT_ORT_LIB_DIR}/libonnxruntime.so")
    elseif(EXISTS "${SPACEMIT_ORT_LIB_DIR}/libonnxruntime.a")
        set(ONNXRUNTIME_LIB "${SPACEMIT_ORT_LIB_DIR}/libonnxruntime.a")
    else()
        message(FATAL_ERROR "Could not find libonnxruntime.{so,a} in SPACEMIT_ORT_DIR='${SPACEMIT_ORT_DIR}'")
    endif()

    target_include_directories(${TARGET} PRIVATE
        ../mtmd
        ${SPACEMIT_ORT_SAMPLE_DIR}
        ${SPACEMIT_ORT_INC_DIR}
    )
    target_link_libraries(${TARGET} PRIVATE ${ONNXRUNTIME_LIB})
    if(CMAKE_DL_LIBS)
        target_link_libraries(${TARGET} PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
