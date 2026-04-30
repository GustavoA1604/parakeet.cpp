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
option(QVAC_PARAKEET_GGML_LIB_PREFIX   "qvac-parakeet: rename bundled ggml libraries to libqvac-parakeet-ggml-* (filename only; CMake target names and C symbols are unchanged). Prevents shared-library filename collisions when multiple addons that vendor different ggml versions are loaded into the same process. Has no effect when QVAC_PARAKEET_USE_SYSTEM_GGML=ON." ON)

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

function(qvac_parakeet_apply_ggml_prefix target)
    if (NOT TARGET ${target})
        return()
    endif()
    get_target_property(_qpgp_type ${target} TYPE)
    if (_qpgp_type STREQUAL "INTERFACE_LIBRARY" OR _qpgp_type STREQUAL "OBJECT_LIBRARY")
        return()
    endif()
    get_target_property(_qpgp_old_name ${target} OUTPUT_NAME)
    if (NOT _qpgp_old_name OR _qpgp_old_name STREQUAL "_qpgp_old_name-NOTFOUND")
        set(_qpgp_old_name ${target})
    endif()
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "qvac-parakeet-${_qpgp_old_name}"
    )
endfunction()

if (QVAC_PARAKEET_GGML_LIB_PREFIX AND NOT QVAC_PARAKEET_USE_SYSTEM_GGML)
    foreach (_qpgp_target ggml ggml-base)
        qvac_parakeet_apply_ggml_prefix(${_qpgp_target})
    endforeach()
    if (DEFINED GGML_AVAILABLE_BACKENDS)
        foreach (_qpgp_target ${GGML_AVAILABLE_BACKENDS})
            qvac_parakeet_apply_ggml_prefix(${_qpgp_target})
        endforeach()
    endif()
    message(STATUS "qvac-parakeet: bundled ggml libraries will be emitted as libqvac-parakeet-ggml-* (set QVAC_PARAKEET_GGML_LIB_PREFIX=OFF to keep upstream filenames)")
endif()

find_package(OpenMP)

# Centralised GGML_USE_* backend defines. Anything that compiles
# parakeet_ctc.cpp (the library target plus the standalone test
# executables that recompile it) must link against this so the
# `init_gpu_backend` / BLAS / CUDA / Metal / Vulkan code paths get
# selected consistently. Without this, e.g. test-encoder would silently
# build with the GPU branch compiled out and `--n-gpu-layers 1` would
# be a no-op.
add_library(qvac-parakeet-backend-defs INTERFACE)
if (GGML_CUDA)
    target_compile_definitions(qvac-parakeet-backend-defs INTERFACE GGML_USE_CUDA)
endif()
if (GGML_METAL)
    target_compile_definitions(qvac-parakeet-backend-defs INTERFACE GGML_USE_METAL)
endif()
if (GGML_VULKAN)
    target_compile_definitions(qvac-parakeet-backend-defs INTERFACE GGML_USE_VULKAN)
endif()
if (GGML_BLAS)
    target_compile_definitions(qvac-parakeet-backend-defs INTERFACE GGML_USE_BLAS)
endif()
if (GGML_OPENCL)
    target_compile_definitions(qvac-parakeet-backend-defs INTERFACE GGML_USE_OPENCL)
endif()

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

    target_link_libraries(qvac-parakeet PRIVATE qvac-parakeet-backend-defs)

    if (GGML_OPENCL)
        target_compile_definitions(qvac-parakeet PRIVATE GGML_USE_OPENCL)
    endif()

    # `PARAKEET_EXPERIMENTAL_FLASH_ATTN` gates the rel-pos MHA
    # fused-attention path in `parakeet_ctc.cpp`. Per-backend defaults:
    #
    #   - Metal:  ON. Validated on M3 Ultra (5 invocations x 15 runs,
    #             sample-16k.wav, q8_0): encoder 67.35 -> 66.96 ms
    #             (-0.39 ms / -0.6 %), inference 119.24 -> 118.84 ms
    #             (-0.40 ms / -0.34 %), parity gates byte-exact (TDT
    #             decoder 95 tokens; encoder-capture 258048 + 258300
    #             floats bit-equal). Closes PROGRESS Section 6.5
    #             "likely a meaningful win on Metal" follow-up.
    #             `kernel_flash_attn_ext_f32_dk128_dv128` is the kernel
    #             actually loaded; head_dim 128 covers parakeet-{ctc,
    #             tdt,eou} d_model=1024/n_heads=8 and parakeet-tdt-1.1b
    #             d_model=2048/n_heads=16. Sortformer's transformer
    #             head_dim=24 falls below ggml_flash_attn_ext's
    #             supported set so the encoder path there is untouched
    #             (Sortformer uses `parakeet_sortformer.cpp` not
    #             `parakeet_ctc.cpp::rel_pos_mha_graph`).
    #   - CPU:    OFF. Cast-to-f16 of the BD mask before softmax
    #             measurably changed accumulator order and showed
    #             +3.1 % encoder regression in Round 7 audit; risks
    #             the bit-equal-NeMo gate.
    #   - Others: OFF until backend-specific A/B. ggml-opencl supports
    #             `flash_attn_ext` for head_dim in
    #             {40,64,80,96,112,128,192,256}; first-Adreno bring-up
    #             should flip and bench. Vulkan validation harness
    #             (tests/test_vk_vs_cpu.cpp) is the natural home for
    #             that A/B once a non-Tegra Vulkan device is available.
    if (GGML_METAL)
        set(QVAC_PARAKEET_FLASH_ATTN_DEFAULT ON)
    else()
        set(QVAC_PARAKEET_FLASH_ATTN_DEFAULT OFF)
    endif()
    option(QVAC_PARAKEET_FLASH_ATTN "qvac-parakeet: enable fused flash-attn in MHA (default ON for Metal; OFF elsewhere pending per-backend A/B)" ${QVAC_PARAKEET_FLASH_ATTN_DEFAULT})
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
        tests/test_mel.cpp
        src/mel_preprocess.cpp
        src/parakeet_ctc.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-mel PRIVATE ggml qvac-parakeet-backend-defs)
    target_include_directories(test-mel PRIVATE ggml/include src include)
    qvac_parakeet_apply_backend_defs(test-mel)

    add_executable(test-encoder
        tests/test_encoder.cpp
        src/parakeet_ctc.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-encoder PRIVATE ggml qvac-parakeet-backend-defs)
    target_include_directories(test-encoder PRIVATE ggml/include src)
    qvac_parakeet_apply_backend_defs(test-encoder)

    add_executable(test-ctc
        tests/test_ctc.cpp
        src/parakeet_ctc.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-ctc PRIVATE ggml qvac-parakeet-backend-defs)
    target_include_directories(test-ctc PRIVATE ggml/include src)
    qvac_parakeet_apply_backend_defs(test-ctc)

    add_executable(test-streaming tests/test_streaming.cpp)
    target_link_libraries(test-streaming PRIVATE qvac-parakeet)
    target_include_directories(test-streaming PRIVATE include src ggml/include)
    qvac_parakeet_apply_ccache(test-streaming)

    add_executable(test-eou-streaming tests/test_eou_streaming.cpp)
    target_link_libraries(test-eou-streaming PRIVATE qvac-parakeet)
    target_include_directories(test-eou-streaming PRIVATE include src ggml/include)
    qvac_parakeet_apply_ccache(test-eou-streaming)

    add_executable(test-sortformer-streaming tests/test_sortformer_streaming.cpp)
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
        tests/test_tdt_encoder_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-tdt-encoder-parity PRIVATE ggml qvac-parakeet-backend-defs)
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
    qvac_parakeet_apply_backend_defs(test-tdt-decoder-parity)

    add_executable(test-sortformer-parity
        tests/test_sortformer_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-sortformer-parity PRIVATE ggml qvac-parakeet-backend-defs)
    target_include_directories(test-sortformer-parity PRIVATE include src ggml/include)
    qvac_parakeet_apply_backend_defs(test-sortformer-parity)

    if (GGML_VULKAN)
        add_executable(test-vk-vs-cpu
            tests/test_vk_vs_cpu.cpp
            src/parakeet_ctc.cpp
            src/mel_preprocess.cpp
            src/sentencepiece_bpe.cpp)
        target_link_libraries(test-vk-vs-cpu PRIVATE ggml qvac-parakeet-backend-defs)
        target_include_directories(test-vk-vs-cpu PRIVATE ggml/include src)
        qvac_parakeet_apply_backend_defs(test-vk-vs-cpu)
    endif()
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
