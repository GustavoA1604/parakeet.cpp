include(CMakeFindDependencyMacro)
find_dependency(ggml CONFIG)

get_filename_component(_PARAKEET_CPP_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

find_library(PARAKEET_LIBRARY
    NAMES parakeet
    PATHS "${_PARAKEET_CPP_PREFIX}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

find_path(PARAKEET_INCLUDE_DIR
    NAMES parakeet/parakeet.h
    PATHS "${_PARAKEET_CPP_PREFIX}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT TARGET parakeet::parakeet)
    add_library(parakeet::parakeet STATIC IMPORTED)
    set_target_properties(parakeet::parakeet PROPERTIES
        IMPORTED_LOCATION             "${PARAKEET_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${PARAKEET_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES      "ggml::ggml"
    )
endif()

unset(_PARAKEET_CPP_PREFIX)
