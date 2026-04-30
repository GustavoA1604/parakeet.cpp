cmake_minimum_required(VERSION 3.14)
project(qvac-parakeet-cpp LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(QVAC_PARAKEET_STANDALONE_DEFAULT ON)
else()
    set(QVAC_PARAKEET_STANDALONE_DEFAULT OFF)
endif()

option(QVAC_PARAKEET_BUILD_LIBRARY     "qvac-parakeet: build the static qvac-parakeet library"                 ON)
option(QVAC_PARAKEET_BUILD_EXECUTABLES "qvac-parakeet: build the qvac-parakeet CLI"                            ${QVAC_PARAKEET_STANDALONE_DEFAULT})
option(QVAC_PARAKEET_BUILD_TESTS       "qvac-parakeet: build per-stage numerical validation harnesses"         ${QVAC_PARAKEET_STANDALONE_DEFAULT})
option(QVAC_PARAKEET_BUILD_EXAMPLES    "qvac-parakeet: build standalone examples (live-mic, ...)"              ${QVAC_PARAKEET_STANDALONE_DEFAULT})
option(QVAC_PARAKEET_BUILD_TOOLS       "qvac-parakeet: build optional dev/CI tools (probe-opencl, ...)"        OFF)
option(QVAC_PARAKEET_INSTALL           "qvac-parakeet: generate install rules"                                 ON)
option(QVAC_PARAKEET_USE_SYSTEM_GGML   "qvac-parakeet: use system-installed GGML library"                      OFF)
option(QVAC_PARAKEET_CCACHE            "qvac-parakeet: use ccache as compiler launcher if available"           ON)

# ccache shaves ~30 % off iterative rebuild time on developer machines.
# Opt-out via -DQVAC_PARAKEET_CCACHE=OFF for clean reproducibility runs.
# Scoped to qvac-parakeet's own targets via the helper below; ggml has
# its own GGML_CCACHE option which we leave alone so the two launchers
# don't double-up on the ggml subdirectory.
if (QVAC_PARAKEET_CCACHE)
    find_program(QVAC_PARAKEET_CCACHE_PROGRAM ccache)
    if (QVAC_PARAKEET_CCACHE_PROGRAM)
        message(STATUS "qvac-parakeet: using ccache at ${QVAC_PARAKEET_CCACHE_PROGRAM} (qvac-parakeet targets only; ggml uses GGML_CCACHE)")
    endif()
endif()

function(qvac_parakeet_apply_ccache target)
    if (QVAC_PARAKEET_CCACHE_PROGRAM)
        set_target_properties(${target} PROPERTIES
            C_COMPILER_LAUNCHER   "${QVAC_PARAKEET_CCACHE_PROGRAM}"
            CXX_COMPILER_LAUNCHER "${QVAC_PARAKEET_CCACHE_PROGRAM}"
        )
    endif()
endfunction()

if (CMAKE_CROSSCOMPILING)
    if (NOT DEFINED GGML_NATIVE OR GGML_NATIVE)
        message(STATUS "qvac-parakeet: cross-compiling to ${CMAKE_SYSTEM_NAME} -- "
                       "forcing GGML_NATIVE=OFF (ggml's ARM feature probes use try_run() "
                       "which CMake refuses in cross-compile mode). The EOU q8_0 precision "
                       "caveat from native macOS/Linux builds does not apply: mobile prebuilds "
                       "are per-arch and use the armv8-a baseline.")
        set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    endif()
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (NOT DEFINED GGML_BLAS OR GGML_BLAS)
        set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    endif()
    if (NOT DEFINED GGML_ACCELERATE OR GGML_ACCELERATE)
        set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
    endif()
endif()

if (NOT TARGET ggml)
    if (QVAC_PARAKEET_USE_SYSTEM_GGML)
        find_package(ggml CONFIG REQUIRED)
        if (NOT ggml_FOUND)
            message(FATAL_ERROR "System-installed GGML library not found.")
        endif()
        add_library(ggml ALIAS ggml::ggml)
    else()
        add_subdirectory(ggml)
    endif()
endif()

find_package(OpenMP)

set(QVAC_PARAKEET_LIB_SOURCES
    src/parakeet_ctc.cpp
    src/parakeet_engine.cpp
    src/parakeet_tdt.cpp
    src/parakeet_eou.cpp
    src/parakeet_sortformer.cpp
    src/mel_preprocess.cpp
    src/sentencepiece_bpe.cpp
    src/energy_vad.cpp
)

if (QVAC_PARAKEET_BUILD_LIBRARY)
    add_library(qvac-parakeet STATIC ${QVAC_PARAKEET_LIB_SOURCES})
    add_library(qvac-parakeet::qvac-parakeet ALIAS qvac-parakeet)
    qvac_parakeet_apply_ccache(qvac-parakeet)

    target_include_directories(qvac-parakeet
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    target_link_libraries(qvac-parakeet PUBLIC ggml)

    if (OpenMP_CXX_FOUND)
        target_link_libraries(qvac-parakeet PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if (GGML_BLAS)
        target_compile_definitions(qvac-parakeet PRIVATE GGML_USE_BLAS)
    endif()
    if (GGML_CUDA)
        target_compile_definitions(qvac-parakeet PRIVATE GGML_USE_CUDA)
    endif()
    if (GGML_METAL)
        target_compile_definitions(qvac-parakeet PRIVATE GGML_USE_METAL)
    endif()
    if (GGML_VULKAN)
        target_compile_definitions(qvac-parakeet PRIVATE GGML_USE_VULKAN)
    endif()
    # GGML_USE_BLAS already surfaced above (matches upstream 8da4939's
    # placement; the OpenCL wire-up originally re-added it before
    # GGML_USE_CUDA which is now redundant after the rebase).
    if (GGML_OPENCL)
        target_compile_definitions(qvac-parakeet PRIVATE GGML_USE_OPENCL)
    endif()

    # `PARAKEET_EXPERIMENTAL_FLASH_ATTN` historically gates the
    # rel-pos MHA fused-attention path in `parakeet_ctc.cpp`. Off by
    # default during the optimization audit because flipping it on CPU
    # measurably changed the rel-pos BD computation order (cast-to-f16
    # mask before softmax) and risked transcript drift on the
    # bit-equal-NeMo gate. Surfaced as an opt-in CMake flag so first
    # Adreno bring-up can A/B it -- ggml-opencl supports
    # `flash_attn_ext` for head_dim in {40,64,80,96,112,128,192,256},
    # which covers every parakeet target except Sortformer's
    # tf_d_model=192/8=24 transformer head.
    option(QVAC_PARAKEET_FLASH_ATTN "qvac-parakeet: enable experimental fused flash-attn in MHA (off by default; bench-validate per backend before shipping)" OFF)
    if (QVAC_PARAKEET_FLASH_ATTN)
        target_compile_definitions(qvac-parakeet PRIVATE PARAKEET_EXPERIMENTAL_FLASH_ATTN)
    endif()

    set_target_properties(qvac-parakeet PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME qvac-parakeet
    )

    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        target_compile_options(qvac-parakeet PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:C>:-O3>
            $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
            $<$<COMPILE_LANGUAGE:C>:-ffast-math>
            $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
            $<$<COMPILE_LANGUAGE:C>:-funroll-loops>
        )
    endif()
endif()

if (QVAC_PARAKEET_BUILD_EXECUTABLES)
    if (NOT QVAC_PARAKEET_BUILD_LIBRARY)
        message(FATAL_ERROR "QVAC_PARAKEET_BUILD_EXECUTABLES requires QVAC_PARAKEET_BUILD_LIBRARY=ON")
    endif()

    add_executable(qvac-parakeet-cli src/cli_main.cpp src/main.cpp)
    target_link_libraries(qvac-parakeet-cli PRIVATE qvac-parakeet)
    target_include_directories(qvac-parakeet-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(qvac-parakeet-cli PROPERTIES OUTPUT_NAME qvac-parakeet)
    qvac_parakeet_apply_ccache(qvac-parakeet-cli)
endif()

if (QVAC_PARAKEET_BUILD_TESTS)
    # Helper: keep the parakeet_ctc.cpp #ifdefs (BLAS / CUDA / Metal / Vulkan /
    # OpenCL backend init) consistent across the qvac-parakeet library and any
    # test executable that compiles parakeet_ctc.cpp from source. Without this,
    # tests that don't link the library would always evaluate the #ifdefs as
    # "no backend defined", producing link errors against ggml-blas / ggml-vk
    # / ggml-opencl when the parent build did enable them.
    function(qvac_parakeet_apply_backend_defs target)
        if (GGML_BLAS)
            target_compile_definitions(${target} PRIVATE GGML_USE_BLAS)
        endif()
        if (GGML_CUDA)
            target_compile_definitions(${target} PRIVATE GGML_USE_CUDA)
        endif()
        if (GGML_METAL)
            target_compile_definitions(${target} PRIVATE GGML_USE_METAL)
        endif()
        if (GGML_VULKAN)
            target_compile_definitions(${target} PRIVATE GGML_USE_VULKAN)
        endif()
        if (GGML_OPENCL)
            target_compile_definitions(${target} PRIVATE GGML_USE_OPENCL)
        endif()
        if (QVAC_PARAKEET_FLASH_ATTN)
            target_compile_definitions(${target} PRIVATE PARAKEET_EXPERIMENTAL_FLASH_ATTN)
        endif()
        qvac_parakeet_apply_ccache(${target})
    endfunction()

    add_executable(test-mel
        src/test_mel.cpp
        src/mel_preprocess.cpp
        src/parakeet_ctc.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-mel PRIVATE ggml)
    target_include_directories(test-mel PRIVATE ggml/include src include)
    qvac_parakeet_apply_backend_defs(test-mel)

    add_executable(test-encoder
        src/test_encoder.cpp
        src/parakeet_ctc.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-encoder PRIVATE ggml)
    target_include_directories(test-encoder PRIVATE ggml/include src)
    qvac_parakeet_apply_backend_defs(test-encoder)

    add_executable(test-ctc
        src/test_ctc.cpp
        src/parakeet_ctc.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-ctc PRIVATE ggml)
    target_include_directories(test-ctc PRIVATE ggml/include src)
    qvac_parakeet_apply_backend_defs(test-ctc)

    add_executable(test-streaming src/test_streaming.cpp)
    target_link_libraries(test-streaming PRIVATE qvac-parakeet)
    target_include_directories(test-streaming PRIVATE include src ggml/include)
    qvac_parakeet_apply_ccache(test-streaming)

    add_executable(test-eou-streaming src/test_eou_streaming.cpp)
    target_link_libraries(test-eou-streaming PRIVATE qvac-parakeet)
    target_include_directories(test-eou-streaming PRIVATE include src ggml/include)
    qvac_parakeet_apply_ccache(test-eou-streaming)

    add_executable(test-sortformer-streaming src/test_sortformer_streaming.cpp)
    target_link_libraries(test-sortformer-streaming PRIVATE qvac-parakeet)
    target_include_directories(test-sortformer-streaming PRIVATE include src ggml/include)
    qvac_parakeet_apply_ccache(test-sortformer-streaming)

    add_executable(test-perf-regression src/test_perf_regression.cpp)
    target_link_libraries(test-perf-regression PRIVATE qvac-parakeet)
    target_include_directories(test-perf-regression PRIVATE include src ggml/include)
    qvac_parakeet_apply_ccache(test-perf-regression)

    # Internal regression gate for the round-2 mel preprocess sweep
    # (real-FFT, MelState reuse). Pure unit test -- no GGUF needed.
    add_executable(test-mel-fft-parity
        src/test_mel_fft_parity.cpp
        src/mel_preprocess.cpp)
    target_link_libraries(test-mel-fft-parity PRIVATE ggml)
    target_include_directories(test-mel-fft-parity PRIVATE ggml/include src include)
    qvac_parakeet_apply_backend_defs(test-mel-fft-parity)

    # Regression gate for the round-1 capture_intermediates flag.
    # Asserts run_encoder(capture=true) and (capture=false) produce
    # bit-equal encoder_out + logits on a CTC GGUF.
    add_executable(test-encoder-capture-parity
        src/test_encoder_capture_parity.cpp
        src/parakeet_ctc.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-encoder-capture-parity PRIVATE ggml)
    target_include_directories(test-encoder-capture-parity PRIVATE ggml/include src include)
    qvac_parakeet_apply_backend_defs(test-encoder-capture-parity)

    add_executable(test-tdt-encoder-parity
        src/test_tdt_encoder_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-tdt-encoder-parity PRIVATE ggml)
    target_include_directories(test-tdt-encoder-parity PRIVATE include src ggml/include)
    qvac_parakeet_apply_backend_defs(test-tdt-encoder-parity)

    add_executable(test-tdt-decoder-parity
        src/test_tdt_decoder_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-tdt-decoder-parity PRIVATE ggml)
    target_include_directories(test-tdt-decoder-parity PRIVATE include src ggml/include)

    add_executable(test-sortformer-parity
        src/test_sortformer_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-sortformer-parity PRIVATE ggml)
    target_include_directories(test-sortformer-parity PRIVATE include src ggml/include)
    qvac_parakeet_apply_backend_defs(test-sortformer-parity)
endif()

if (QVAC_PARAKEET_BUILD_EXAMPLES)
    if (NOT QVAC_PARAKEET_BUILD_LIBRARY)
        message(FATAL_ERROR "QVAC_PARAKEET_BUILD_EXAMPLES requires QVAC_PARAKEET_BUILD_LIBRARY=ON")
    endif()

    add_executable(live-mic
        examples/live-mic.cpp
        examples/miniaudio_impl.cpp)
    target_link_libraries(live-mic PRIVATE qvac-parakeet)
    target_include_directories(live-mic PRIVATE include examples)
    target_compile_options(live-mic PRIVATE -Wno-deprecated-declarations)
    qvac_parakeet_apply_ccache(live-mic)

    add_executable(live-mic-attributed
        examples/live-mic-attributed.cpp
        examples/miniaudio_impl.cpp)
    target_link_libraries(live-mic-attributed PRIVATE qvac-parakeet)
    target_include_directories(live-mic-attributed PRIVATE include examples)
    target_compile_options(live-mic-attributed PRIVATE -Wno-deprecated-declarations)
    qvac_parakeet_apply_ccache(live-mic-attributed)

    if (APPLE)
        find_library(COREAUDIO_LIBRARY CoreAudio REQUIRED)
        find_library(COREFOUNDATION_LIBRARY CoreFoundation REQUIRED)
        find_library(AUDIOUNIT_LIBRARY AudioUnit REQUIRED)
        find_library(AUDIOTOOLBOX_LIBRARY AudioToolbox REQUIRED)
        foreach (tgt live-mic live-mic-attributed)
            target_link_libraries(${tgt} PRIVATE
                ${COREAUDIO_LIBRARY}
                ${COREFOUNDATION_LIBRARY}
                ${AUDIOUNIT_LIBRARY}
                ${AUDIOTOOLBOX_LIBRARY})
        endforeach()
    elseif (UNIX)
        find_package(Threads REQUIRED)
        foreach (tgt live-mic live-mic-attributed)
            target_link_libraries(${tgt} PRIVATE Threads::Threads dl m)
        endforeach()
    elseif (WIN32)
        foreach (tgt live-mic live-mic-attributed)
            target_link_libraries(${tgt} PRIVATE winmm ole32)
        endforeach()
    endif()
endif()

if (QVAC_PARAKEET_BUILD_TOOLS)
    if (NOT GGML_OPENCL)
        message(STATUS "qvac-parakeet: QVAC_PARAKEET_BUILD_TOOLS requested but GGML_OPENCL=OFF; skipping probe-opencl")
    else()
        find_package(OpenCL QUIET)
        if (NOT OpenCL_FOUND)
            message(WARNING "qvac-parakeet: QVAC_PARAKEET_BUILD_TOOLS requested but OpenCL headers/loader not found; skipping probe-opencl")
        else()
            add_executable(probe-opencl scripts/probe-opencl.cpp)
            target_link_libraries(probe-opencl PRIVATE OpenCL::OpenCL)
            target_compile_definitions(probe-opencl PRIVATE CL_TARGET_OPENCL_VERSION=300)
            qvac_parakeet_apply_ccache(probe-opencl)
        endif()
    endif()
endif()

if (QVAC_PARAKEET_INSTALL AND QVAC_PARAKEET_BUILD_LIBRARY)
    install(TARGETS qvac-parakeet
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/qvac-parakeet
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/qvac-parakeet-cppConfig.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/qvac-parakeet-cppConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_DATADIR}/qvac-parakeet-cpp
    )
endif()
