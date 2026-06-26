include(GNUInstallDirs)

set(Q3TTS_TARGET_PREFIX llama-q3tts)
set(Q3TTS_SCRIPT_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
if(NOT Q3TTS_SCRIPT_OUTPUT_DIR)
    set(Q3TTS_SCRIPT_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(Q3TTS_ONNXRUNTIME_ROOT "" CACHE PATH "Root of an ONNX Runtime installation used by Qwen3-TTS")
set(Q3TTS_SPACEMIT_EP_ROOT "" CACHE PATH "Root of a SpacemiT EP installation copied into q3tts runtime packages")
option(Q3TTS_COPY_RUNTIME_LIBS "Copy ONNX Runtime and SpacemiT EP shared libraries into q3tts-install prefix" ON)

set(Q3TTS_ONNXRUNTIME_INCLUDE_HINTS)
set(Q3TTS_ONNXRUNTIME_LIB_HINTS)
foreach(_q3tts_ort_root IN ITEMS "${Q3TTS_ONNXRUNTIME_ROOT}" "$ENV{Q3TTS_ONNXRUNTIME_ROOT}")
    if(_q3tts_ort_root)
        list(APPEND Q3TTS_ONNXRUNTIME_INCLUDE_HINTS
            ${_q3tts_ort_root}/include
            ${_q3tts_ort_root}/include/onnxruntime
            ${_q3tts_ort_root}/include/onnxruntime/core/session
        )
        list(APPEND Q3TTS_ONNXRUNTIME_LIB_HINTS
            ${_q3tts_ort_root}/lib
            ${_q3tts_ort_root}/lib64
        )
    endif()
endforeach()

find_path(Q3TTS_ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${Q3TTS_ONNXRUNTIME_INCLUDE_HINTS}
    NO_DEFAULT_PATH
)
find_library(Q3TTS_ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${Q3TTS_ONNXRUNTIME_LIB_HINTS}
    NO_DEFAULT_PATH
)
if(NOT Q3TTS_ONNXRUNTIME_INCLUDE_DIR)
    find_path(Q3TTS_ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
        /usr/include
        /usr/include/onnxruntime
        /usr/include/onnxruntime/core/session
        /usr/local/include
        /usr/local/include/onnxruntime
        /usr/local/include/onnxruntime/core/session
    )
endif()
if(NOT Q3TTS_ONNXRUNTIME_LIB)
    find_library(Q3TTS_ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS
        /usr/lib
        /usr/local/lib
    )
endif()

set(Q3TTS_SPACEMIT_EP_LIB_HINTS)
foreach(_q3tts_ep_root IN ITEMS "${Q3TTS_SPACEMIT_EP_ROOT}" "$ENV{Q3TTS_SPACEMIT_EP_ROOT}")
    if(_q3tts_ep_root)
        list(APPEND Q3TTS_SPACEMIT_EP_LIB_HINTS
            ${_q3tts_ep_root}/lib
            ${_q3tts_ep_root}/lib64
        )
    endif()
endforeach()
find_library(Q3TTS_SPACEMIT_EP_LIB
    NAMES spacemit_ep
    PATHS ${Q3TTS_SPACEMIT_EP_LIB_HINTS}
    NO_DEFAULT_PATH
)
if(NOT Q3TTS_SPACEMIT_EP_LIB)
    find_library(Q3TTS_SPACEMIT_EP_LIB
        NAMES spacemit_ep
        PATHS /usr/lib /usr/local/lib
    )
endif()

if(NOT Q3TTS_ONNXRUNTIME_INCLUDE_DIR OR NOT Q3TTS_ONNXRUNTIME_LIB)
    message(STATUS "Qwen3-TTS tools disabled; ONNX Runtime headers/library not found")
    return()
endif()
message(STATUS "Qwen3-TTS ONNX Runtime include: ${Q3TTS_ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Qwen3-TTS ONNX Runtime library: ${Q3TTS_ONNXRUNTIME_LIB}")
if(Q3TTS_SPACEMIT_EP_LIB)
    message(STATUS "Qwen3-TTS SpacemiT EP library: ${Q3TTS_SPACEMIT_EP_LIB}")
else()
    message(STATUS "Qwen3-TTS SpacemiT EP library not found; runtime will rely on system lookup if needed")
endif()

set(Q3TTS_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/qwen3_tts
    ${Q3TTS_ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)

add_library(${Q3TTS_TARGET_PREFIX}-runtime STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen3_tts_runtime.cpp
)
target_include_directories(${Q3TTS_TARGET_PREFIX}-runtime
    PUBLIC
        ${Q3TTS_INCLUDE_DIRS}
)
target_link_libraries(${Q3TTS_TARGET_PREFIX}-runtime
    PUBLIC
        ${Q3TTS_ONNXRUNTIME_LIB}
        ggml-base
    PRIVATE
        dl
        pthread
        m
)
target_compile_features(${Q3TTS_TARGET_PREFIX}-runtime PRIVATE cxx_std_17)

find_path(Q3TTS_ALSA_INCLUDE_DIR
    NAMES alsa/asoundlib.h
    PATHS /usr/include /usr/local/include
)
find_library(Q3TTS_ALSA_LIB
    NAMES asound
    PATHS /usr/lib /usr/local/lib
)
if(Q3TTS_ALSA_INCLUDE_DIR AND Q3TTS_ALSA_LIB)
    target_compile_definitions(${Q3TTS_TARGET_PREFIX}-runtime PUBLIC Q3TTS_ENABLE_SDK_AUDIO)
    target_include_directories(${Q3TTS_TARGET_PREFIX}-runtime PUBLIC ${Q3TTS_ALSA_INCLUDE_DIR})
    target_link_libraries(${Q3TTS_TARGET_PREFIX}-runtime PUBLIC ${Q3TTS_ALSA_LIB})
endif()

add_executable(q3tts-runner ${CMAKE_CURRENT_SOURCE_DIR}/tools/q3tts_run_main.cpp)
target_link_libraries(q3tts-runner PRIVATE ${Q3TTS_TARGET_PREFIX}-runtime)
target_compile_features(q3tts-runner PRIVATE cxx_std_17)

add_executable(q3tts-ref-to-bin ${CMAKE_CURRENT_SOURCE_DIR}/tools/q3tts_ref_to_bin.cpp)
target_include_directories(q3tts-ref-to-bin PRIVATE ${Q3TTS_INCLUDE_DIRS})
target_link_libraries(q3tts-ref-to-bin PRIVATE
    ${Q3TTS_ONNXRUNTIME_LIB}
    ggml-base
    pthread
    m
)
target_compile_features(q3tts-ref-to-bin PRIVATE cxx_std_17)

add_executable(talker_driver.headmain ${CMAKE_CURRENT_SOURCE_DIR}/src/talker_driver.c)
target_compile_definitions(talker_driver.headmain PRIVATE _GNU_SOURCE)
target_compile_options(talker_driver.headmain PRIVATE
    $<$<C_COMPILER_ID:GNU>:-O2>
    $<$<C_COMPILER_ID:GNU>:-fno-tree-vectorize>
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv)")
    target_compile_options(talker_driver.headmain PRIVATE
        $<$<C_COMPILER_ID:GNU>:-march=rv64gcv_zfh_zvfh_zba_zicbop_zihintpause>
        $<$<C_COMPILER_ID:GNU>:-mabi=lp64d>
    )
endif()
target_include_directories(talker_driver.headmain PRIVATE
    ${Q3TTS_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
)
target_link_libraries(talker_driver.headmain PRIVATE
    llama
    ggml-cpu
    ggml-base
    ggml
    pthread
    m
)

add_executable(q3tts-cp-kernel-bench ${CMAKE_CURRENT_SOURCE_DIR}/tools/q3tts_cp_kernel_bench.cpp)
target_include_directories(q3tts-cp-kernel-bench PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu
)
target_link_libraries(q3tts-cp-kernel-bench PRIVATE
    ggml-cpu
    ggml-base
    ggml
    pthread
    m
)
target_compile_features(q3tts-cp-kernel-bench PRIVATE cxx_std_17)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/q3tts-run.in
               ${Q3TTS_SCRIPT_OUTPUT_DIR}/q3tts-run @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/talker_driver.qwen3tts-k3.in
               ${Q3TTS_SCRIPT_OUTPUT_DIR}/talker_driver.qwen3tts-k3 @ONLY)

execute_process(COMMAND chmod +x
    ${Q3TTS_SCRIPT_OUTPUT_DIR}/q3tts-run
    ${Q3TTS_SCRIPT_OUTPUT_DIR}/talker_driver.qwen3tts-k3
)

get_filename_component(Q3TTS_ONNXRUNTIME_LIB_DIR ${Q3TTS_ONNXRUNTIME_LIB} DIRECTORY)
set(Q3TTS_ONNXRUNTIME_RUNTIME_LIBS)
if(Q3TTS_COPY_RUNTIME_LIBS)
    file(GLOB Q3TTS_ONNXRUNTIME_RUNTIME_LIBS
        ${Q3TTS_ONNXRUNTIME_LIB_DIR}/libonnxruntime.so*
    )
endif()
set(Q3TTS_SPACEMIT_EP_RUNTIME_LIBS)
if(Q3TTS_COPY_RUNTIME_LIBS AND Q3TTS_SPACEMIT_EP_LIB)
    get_filename_component(Q3TTS_SPACEMIT_EP_LIB_DIR ${Q3TTS_SPACEMIT_EP_LIB} DIRECTORY)
    file(GLOB Q3TTS_SPACEMIT_EP_RUNTIME_LIBS
        ${Q3TTS_SPACEMIT_EP_LIB_DIR}/libspacemit_ep.so*
    )
endif()
set(Q3TTS_RUNTIME_LIB_COPY_COMMANDS)
foreach(_q3tts_runtime_lib IN LISTS Q3TTS_ONNXRUNTIME_RUNTIME_LIBS Q3TTS_SPACEMIT_EP_RUNTIME_LIBS)
    list(APPEND Q3TTS_RUNTIME_LIB_COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_q3tts_runtime_lib}
            ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/
    )
endforeach()
set(Q3TTS_BUILD_RUNTIME_LIB_COPY_COMMANDS)
if(Q3TTS_COPY_RUNTIME_LIBS)
    set(Q3TTS_BUILD_RUNTIME_LIB_DIR ${Q3TTS_SCRIPT_OUTPUT_DIR}/../${CMAKE_INSTALL_LIBDIR})
    list(APPEND Q3TTS_BUILD_RUNTIME_LIB_COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E make_directory ${Q3TTS_BUILD_RUNTIME_LIB_DIR}
    )
    foreach(_q3tts_runtime_lib IN LISTS Q3TTS_ONNXRUNTIME_RUNTIME_LIBS Q3TTS_SPACEMIT_EP_RUNTIME_LIBS)
        list(APPEND Q3TTS_BUILD_RUNTIME_LIB_COPY_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_q3tts_runtime_lib}
                ${Q3TTS_BUILD_RUNTIME_LIB_DIR}/
        )
    endforeach()
endif()
if(Q3TTS_BUILD_RUNTIME_LIB_COPY_COMMANDS)
    add_custom_command(TARGET q3tts-runner POST_BUILD
        ${Q3TTS_BUILD_RUNTIME_LIB_COPY_COMMANDS}
        VERBATIM
    )
endif()

set(Q3TTS_LLAMA_SERVER_COPY_COMMANDS)
set(Q3TTS_LLAMA_SERVER_DEPENDS)
if(TARGET llama-server)
    list(APPEND Q3TTS_LLAMA_SERVER_COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:llama-server>
            ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/llama-server
    )
    list(APPEND Q3TTS_LLAMA_SERVER_DEPENDS llama-server)
endif()

add_custom_target(q3tts-install
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:llama> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_FILE_NAME:llama>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_SONAME_FILE:llama> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_SONAME_FILE_NAME:llama>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_LINKER_FILE:llama> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_LINKER_FILE_NAME:llama>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml-cpu> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_FILE_NAME:ggml-cpu>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_SONAME_FILE:ggml-cpu> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_SONAME_FILE_NAME:ggml-cpu>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_LINKER_FILE:ggml-cpu> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_LINKER_FILE_NAME:ggml-cpu>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml-base> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_FILE_NAME:ggml-base>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_SONAME_FILE:ggml-base> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_SONAME_FILE_NAME:ggml-base>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_LINKER_FILE:ggml-base> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_LINKER_FILE_NAME:ggml-base>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_FILE_NAME:ggml>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_SONAME_FILE:ggml> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_SONAME_FILE_NAME:ggml>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_LINKER_FILE:ggml> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_LINKER_FILE_NAME:ggml>
    ${Q3TTS_RUNTIME_LIB_COPY_COMMANDS}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:q3tts-runner> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/q3tts-runner
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:q3tts-ref-to-bin> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/q3tts-ref-to-bin
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:talker_driver.headmain> ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/talker_driver.headmain
    ${Q3TTS_LLAMA_SERVER_COPY_COMMANDS}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Q3TTS_SCRIPT_OUTPUT_DIR}/q3tts-run ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/q3tts-run
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Q3TTS_SCRIPT_OUTPUT_DIR}/talker_driver.qwen3tts-k3 ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/talker_driver.qwen3tts-k3
    COMMAND chmod +x
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/q3tts-run
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/talker_driver.qwen3tts-k3
    DEPENDS q3tts-runner q3tts-ref-to-bin talker_driver.headmain ${Q3TTS_LLAMA_SERVER_DEPENDS}
    VERBATIM
)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS q3tts-runner q3tts-ref-to-bin talker_driver.headmain RUNTIME)
    install(PROGRAMS
        ${Q3TTS_SCRIPT_OUTPUT_DIR}/q3tts-run
        ${Q3TTS_SCRIPT_OUTPUT_DIR}/talker_driver.qwen3tts-k3
        TYPE BIN
    )
endif()
