#pragma once

// Top-level QVAC Parakeet library entry points.
//
// The library currently ships the Parakeet-CTC English pipeline.  Additional
// engines (TDT multilingual, Sortformer diarization, EOU streaming) will land
// under the same umbrella and should prefer headers under <qvac-parakeet/...>
// for the generic API and <qvac-parakeet/<engine>/...> for engine-specific
// details.
//
// Two layers of API are exposed today:
//
//   1. High-level wav -> text via the CLI dispatcher.  The current
//      implementation wraps the CLI's argv path; a proper struct-based
//      public API will land as the code is split out of src/main.cpp.
//      Until then, callers building against the library can invoke
//      `qvac_parakeet_cli_main(argc, argv)` with the same flags accepted by
//      the `qvac-parakeet` executable.
//
//   2. Lower-level per-engine APIs, e.g. the Parakeet-CTC pipeline in
//      <qvac-parakeet/ctc/pipeline.h>.

#ifdef __cplusplus
extern "C" {
#endif

int qvac_parakeet_cli_main(int argc, char ** argv);

#ifdef __cplusplus
}
#endif
