#pragma once

// CLI entry point reused by the qvac-parakeet executable.
//
// `qvac_parakeet_cli_main(argc, argv)` accepts the same flag set as the
// `qvac-parakeet` binary (run with `--help` for the full surface).
// Embedders can wire it into their own driver -- e.g. a unit-test
// harness, an Electron main-process spawner, or a wrapper executable
// -- without rebuilding the parser.
//
// Pure C ABI (extern "C", returns int, takes argv); compatible with
// dlsym / dlopen surfaces that don't speak C++ name mangling.

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

QVAC_PARAKEET_API int qvac_parakeet_cli_main(int argc, char ** argv);

#ifdef __cplusplus
}
#endif
