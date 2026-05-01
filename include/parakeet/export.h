#pragma once

// Symbol-visibility decorator for parakeet's public surface.
//
// Mirrors the LLAMA_API / WHISPER_API macros from sibling ggml-family
// libraries: when parakeet is built as a shared object, public
// entry points are tagged with the platform-correct
// `__declspec(dllexport)` / `__attribute__((visibility("default")))`
// while everything else (the Engine internals, parakeet_ctc loader,
// per-decoder helpers, ...) defaults to hidden via
// `CXX_VISIBILITY_PRESET=hidden` + `VISIBILITY_INLINES_HIDDEN=ON` in
// the consuming target. When built as a static library (the default
// in standalone builds today and the only mode the QVAC vcpkg port
// ships) the macro expands to nothing -- visibility attributes have
// no effect on static archives.
//
// Build flags:
//   - `PARAKEET_SHARED` (PUBLIC define): set when building a
//     shared library. Consumers building against a shared
//     libparakeet must also define this so the symbols import
//     correctly on Windows.
//   - `PARAKEET_BUILD`  (PRIVATE define): set inside the
//     library's own translation units when compiling the shared
//     library. Flips the Windows path from `dllimport` to
//     `dllexport`. CMakeLists.txt sets both PUBLIC and PRIVATE
//     defines automatically when `BUILD_SHARED_LIBS=ON`.

#ifdef PARAKEET_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef PARAKEET_BUILD
#      define PARAKEET_API __declspec(dllexport)
#    else
#      define PARAKEET_API __declspec(dllimport)
#    endif
#  else
#    define PARAKEET_API __attribute__((visibility("default")))
#  endif
#else
#  define PARAKEET_API
#endif
