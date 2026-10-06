# Quality checks

The suite uses Catch2 with exact golden math, generated invariants and real-binary
CLI checks. No model services or GPUs are called. CMake configuration may download
hash-pinned dependencies; test execution is offline. Project warnings are enabled
and treated as errors in presets/CI; third-party sources do not inherit project
warning flags. Code style uses clang-format's Google base at 100 columns.

`python3 scripts/analyze.py --build build-dev` runs focused clang-tidy and exhaustive
cppcheck on `src/`. `.clang-tidy` selects bugprone, cert, performance, a
cppcoreguidelines subset and readability redundancy checks. The parameter-swap
heuristic is disabled because checked arithmetic intentionally accepts symmetric
same-type operands; semantic arithmetic tests cover those operations. This is a
focused check set, not a claim of exhaustive formal analysis.

Coverage instruments `src/`, including the CLI. Illustrative GCC commands:
`cmake --preset coverage`, `cmake --build --preset coverage`, `ctest --preset coverage`,
then `gcovr --root . --filter 'src/' --fail-under-line 90 build-coverage`.
On AppleClang, use `--gcov-executable 'xcrun llvm-cov gcov'`.
The 90% gate is line coverage; branch coverage is separately reported, not conflated.
Local macOS path-access restrictions required an exact temporary source copy for
coverage collection; no source files were excluded to make the number pass.

`asan-ubsan` enables both sanitizers and disables UBSan recovery. The local Apple
ASan runtime hung even on a trivial executable; this is unverified, not a passing
sanitizer result. A separate full UBSan run is used to distinguish that runtime
issue from undefined behavior checks. The combined job is configured for Linux CI,
which has not run for this branch and must be verified after publication.

## Fuzzing

`cmake --preset fuzz` uses Clang/libFuzzer plus non-recovering UBSan, with normal
tests disabled for that build. Config and vLLM-log targets catch only documented
input/arithmetic errors. Unexpected exceptions, crashes and sanitizer failures
remain failures. Seeds are under `fuzz/corpus/`; use a copied temporary corpus,
so fuzz mutations never alter committed seeds. Example, illustrative:

```sh
cmake --preset fuzz
cmake --build --preset fuzz
cp -R fuzz/corpus/config /tmp/kvbench-config-corpus
./build-fuzz/fuzz_config /tmp/kvbench-config-corpus -max_total_time=10 -max_len=16384 -timeout=3
```

Choose a fresh temporary path on your platform. `fuzz_vllm` takes the log corpus
similarly. The local Apple compiler lacks a bundled libFuzzer runtime; a temporary
LLVM 18.1.8 runtime was built from upstream source and supplied through
`-DKVBENCH_FUZZ_RUNTIME=/path/libFuzzer.a`. Standard Linux Clang needs no override.
Bounded fuzzing does not prove immunity to every input or resource attack.

## Local P8 results

The full normal and separate non-recovering UBSan suites passed 51 CTest tests.
Coverage on an exact source copy: 1364/1442 src lines = 94.59%; reported branch
coverage is 52.3% (not the line gate). A rebuild initially had incompatible old
GCDA counters; deleting only generated counters in the temporary build and
rerunning the same full suite produced the measurement above.

Selected clang-tidy checks and exhaustive cppcheck reported no project findings.
Tools: clang-tidy 22.1.8, cppcheck 2.17.1, gcovr 8.6, clang-format 23.1.2,
AppleClang 17.0.0. With upstream LLVM 18.1.8 libFuzzer and UBSan, bounded runs
completed 58,717 config inputs and 2,418 log inputs in 11 seconds each, without
crashes or sanitizer failures. Seed 1729, max_len=16384 and per-input timeout=3.
These are bounded local results, not exhaustive coverage or a remote CI pass.
ASan still hangs at runtime startup even for a trivial program; it is unverified.
C++ mutation testing was not run; no score is claimed.
