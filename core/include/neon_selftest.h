#pragma once
#include <string>

namespace neon_selftest {

// Runs matmul_int8_scalar and (if compiled with NEON) matmul_int8_neon on
// several synthetic inputs of varying size (including sizes NOT divisible
// by 8, to exercise the NEON remainder-loop path), and compares outputs.
// Returns a human-readable report; the app displays this directly so the
// person running it can see PASS/FAIL and the actual max error, rather
// than just trusting the code compiled without complaint.
std::string run();

} // namespace neon_selftest
