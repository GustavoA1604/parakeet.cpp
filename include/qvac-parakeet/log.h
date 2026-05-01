#pragma once

// Host-supplied log sink for libqvac-parakeet.
//
// `qvac_parakeet_log_set` mirrors `llama_log_set` from the
// qvac-fabric-llm.cpp sibling so addons can demux every (level, text)
// emission from libqvac-parakeet into their own structured logging
// without scraping stderr. Pass `cb = nullptr` to revert to the default
// sink (stderr).
//
// The signature is `ggml_log_callback`, the same one `ggml_log_set`
// already accepts, so a single host shim can fan one log pipe out
// across both libraries (and the bundled ggml itself). Default is no
// callback installed (stderr fallback), matching pre-shim behaviour
// byte-for-byte.
//
// `ggml.h` is intentionally exposed in this header: the callback
// signature is owned by ggml, hosts wiring up logging will have ggml
// available anyway, and decoupling the signature would force every
// consumer to write a trampoline. Same shape as whisper.cpp / llama.cpp.

#include "export.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

QVAC_PARAKEET_API void qvac_parakeet_log_set(ggml_log_callback cb,
                                             void *            user_data);

#ifdef __cplusplus
}
#endif
