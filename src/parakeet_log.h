#pragma once

// Internal logging shim for qvac-parakeet.
//
// Routes diagnostic output through a host-supplied `ggml_log_callback`
// when one has been installed via `qvac_parakeet_log_set` (declared in
// the public umbrella header). When no callback is installed the
// shim falls through to `std::fprintf(stderr, ...)` so existing
// standalone CLI / examples behave exactly as before.
//
// Mirrors the shape `qvac-fabric-llm.cpp` (llama_log_set) ships --
// addons hosting libqvac-parakeet alongside libllama can pin one
// callback per library and demux the output into structured logs
// (level + text), with no buffering or locking inside the library.

#include "ggml.h"

#ifdef __GNUC__
#define PARAKEET_LOG_PRINTF_ATTR(fmt_idx, vargs_idx) \
    __attribute__((format(printf, fmt_idx, vargs_idx)))
#else
#define PARAKEET_LOG_PRINTF_ATTR(fmt_idx, vargs_idx)
#endif

namespace qvac_parakeet {

void log_impl(enum ggml_log_level level, const char * fmt, ...) PARAKEET_LOG_PRINTF_ATTR(2, 3);

void log_set_callback(ggml_log_callback cb, void * user_data);

}

#define PARAKEET_LOG(level, ...) ::qvac_parakeet::log_impl((level), __VA_ARGS__)
#define PARAKEET_LOG_DEBUG(...)  PARAKEET_LOG(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define PARAKEET_LOG_INFO(...)   PARAKEET_LOG(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define PARAKEET_LOG_WARN(...)   PARAKEET_LOG(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define PARAKEET_LOG_ERROR(...)  PARAKEET_LOG(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
