include(CMakeFindDependencyMacro)
find_dependency(ggml CONFIG)

get_filename_component(_QVAC_PARAKEET_CPP_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

find_library(QVAC_PARAKEET_LIBRARY
    NAMES qvac-parakeet
    PATHS "${_QVAC_PARAKEET_CPP_PREFIX}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

find_path(QVAC_PARAKEET_INCLUDE_DIR
    NAMES qvac-parakeet/qvac-parakeet.h
    PATHS "${_QVAC_PARAKEET_CPP_PREFIX}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT TARGET qvac-parakeet::qvac-parakeet)
    add_library(qvac-parakeet::qvac-parakeet STATIC IMPORTED)
    set_target_properties(qvac-parakeet::qvac-parakeet PROPERTIES
        IMPORTED_LOCATION             "${QVAC_PARAKEET_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${QVAC_PARAKEET_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES      "ggml::ggml"
    )
endif()

unset(_QVAC_PARAKEET_CPP_PREFIX)
