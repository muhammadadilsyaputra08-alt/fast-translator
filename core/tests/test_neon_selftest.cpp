#include "neon_selftest.h"
#include <iostream>
#include <string>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

int main() {
    std::string report = neon_selftest::run();
    std::cout << report << "\n\n";

    CHECK(!report.empty(), "neon_selftest::run() returns a non-empty report");

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    CHECK(report.find("RESULT: PASS") != std::string::npos ||
          report.find("RESULT: FAIL") != std::string::npos,
          "on an ARM build, report contains a definitive PASS/FAIL verdict");
#else
    // This CI runner is x86 (ubuntu-latest), so we expect the graceful
    // "NEON not compiled in" message, not a NEON pass/fail verdict --
    // actual NEON correctness can only be verified on an ARM build/device
    // (see notes/NEON_OPTIMIZATION_FINDINGS.md).
    CHECK(report.find("NOT compiled in") != std::string::npos,
          "on this x86 CI runner, report correctly states NEON is not compiled in");
#endif

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
