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
option(QVAC_PARAKEET_INSTALL           "qvac-parakeet: generate install rules"                                 ON)
option(QVAC_PARAKEET_USE_SYSTEM_GGML   "qvac-parakeet: use system-installed GGML library"                      OFF)

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
    src/mel_preprocess.cpp
    src/sentencepiece_bpe.cpp
)

if (QVAC_PARAKEET_BUILD_LIBRARY)
    add_library(qvac-parakeet STATIC ${QVAC_PARAKEET_LIB_SOURCES})
    add_library(qvac-parakeet::qvac-parakeet ALIAS qvac-parakeet)

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

    if (GGML_CUDA)
        target_compile_definitions(qvac-parakeet PRIVATE GGML_USE_CUDA)
    endif()
    if (GGML_METAL)
        target_compile_definitions(qvac-parakeet PRIVATE GGML_USE_METAL)
    endif()
    if (GGML_VULKAN)
        target_compile_definitions(qvac-parakeet PRIVATE GGML_USE_VULKAN)
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
endif()

if (QVAC_PARAKEET_BUILD_TESTS)
    add_executable(test-mel
        src/test_mel.cpp
        src/mel_preprocess.cpp
        src/parakeet_ctc.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-mel PRIVATE ggml)
    target_include_directories(test-mel PRIVATE ggml/include src include)

    add_executable(test-encoder
        src/test_encoder.cpp
        src/parakeet_ctc.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-encoder PRIVATE ggml)
    target_include_directories(test-encoder PRIVATE ggml/include src)

    add_executable(test-ctc
        src/test_ctc.cpp
        src/parakeet_ctc.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-ctc PRIVATE ggml)
    target_include_directories(test-ctc PRIVATE ggml/include src)
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
