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
issue from undefined behavior checks. The combined job also runs on Linux CI; its verified remote result is recorded
below separately from the macOS-local limitation.

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

## Local P11 results

Final development, Release, coverage and separate non-recovering UBSan runs each
passed all 58 CTest cases. Catch2 reports 1203 assertions in 51 test cases; the
remaining CTest checks exercise the CLI, examples, documentation, packaging and
workflow policy. The golden subset passed 48 assertions in five cases.

Final src line coverage: 1389/1454 = 95.53%; reported branch
coverage 53.2% (not a gate). The source copy includes every src file,
and generated GCDA counters were cleared before the full measurement. Selected
clang-tidy and exhaustive cppcheck passed with no project findings; clang-format
checks passed. Fresh bounded fuzz runs completed 110,954 config inputs and 3,043
log inputs in 11 seconds each, with no crashes or UBSan failures; flags and
runtime are the same as P8. CPU benchmark results were rerun in Release and
are recorded in [performance](performance.md).

The combined ASan/UBSan build completed after deferring Catch2 discovery to
test time (`-DCMAKE_CATCH_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST`). Test discovery
still hung at local ASan runtime startup; the bounded 10-second attempt was
stopped. ASan execution remains UNVERIFIED. Remote compiler/scanner/sanitizer
jobs and real-engine calibration remain pending. Mutation testing was skipped.

## Raw and filtered branch reporting

The raw branch figure is retained. C++ coverage can count compiler-generated
exception/unwinding edges from operations that may throw as well as source-level
decisions. Such edges increase the denominator, and many are not exercised by
ordinary success paths. gcovr's classification is compiler/coverage-format
dependent; filtering does not turn this into proof of decision coverage.

CI also reports a separate figure using
`--exclude-throw-branches --exclude-unreachable-branches`, with separate JSON/text
artifacts. Neither branch figure is gated. On the same local P11 counters, raw
coverage is 2275/4278 = 53.2%; filtered coverage is 2260/4244 = 53.3%. This small
change is the actual AppleClang result; it is not replaced by a more flattering
number. The line gate remains 90% and both branch figures remain visible.

The Linux ASan+UBSan job runs the full CTest preset with
`ASAN_OPTIONS=detect_leaks=1:abort_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`, including test discovery/build.
The observed ASan hang is a macOS-local limitation until Linux evidence says
otherwise; the Linux job is not skipped or reduced.

## Verified remote results — 2026-10-06

Measured implementation: `3d84eb2`. All eight jobs passed in
[CI run 37475033670](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033670),
and [CodeQL run 37475033616](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033616) passed.

| Job | Result | Evidence |
| --- | --- | --- |
| Linux GCC | Passed full 58-test suite | [job](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033670/job/112308118660) |
| Linux Clang | Passed full 58-test suite | [job](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033670/job/112308118179) |
| macOS AppleClang | Passed full 58-test suite | [job](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033670/job/112308117988) |
| Windows MSVC | Passed full 58-test suite with /W4 /WX | [job](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033670/job/112308118096) |
| Linux ASan+UBSan | Passed full 58-test suite; leak detection enabled | [job](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033670/job/112308118181) |
| Coverage | Passed full suite and 90% src line gate; both branch reports uploaded | [job](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033670/job/112308117965) |
| clang-tidy/cppcheck | Passed selected checks | [job](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033670/job/112308117736) |
| Fuzz smoke | Passed both bounded targets | [job](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033670/job/112308117759) |
| CodeQL C++ | Passed build and analysis | [job](https://github.com/vedansh-adepu/kvbench/actions/runs/37475033616/job/112308115514) |

Compiler/full-suite jobs include generated-results drift, README examples, CLI,
packaging and workflow checks. Local Catch2 counts remain 51 cases and 1203
assertions. Linux sanitizer output reports 58/58 CTest cases passed in 3.37 s;
it did not reproduce the local macOS ASan startup hang. No sanitizer job was
dropped and no compiler warnings were weakened.

Actual Linux GCC/gcovr src measurements from the uploaded artifacts:

| Metric | Covered / total | Percent | Gate |
| --- | ---: | ---: | --- |
| Lines | 1260 / 1320 | 95.45% | 90% |
| Raw branches | 2629 / 5316 | 49.45% | None |
| Filtered branches | 1989 / 2359 | 84.31% | None |

The filtered command uses both `--exclude-throw-branches` and
`--exclude-unreachable-branches`. Raw figures are retained; differences from
AppleClang reflect compiler instrumentation/classification, not source exclusion.
No branch gate was added. Linux fuzz smoke completed 59,855 config inputs and
75 log inputs in 11 seconds each, without crashes; these bounded counts are
host/run dependent and do not measure coverage or prove exhaustive safety.

The initial [CI](https://github.com/vedansh-adepu/kvbench/actions/runs/37474564117)
and [CodeQL](https://github.com/vedansh-adepu/kvbench/actions/runs/37474564113)
builds failed on GCC's conversion warning for invalid-enum test value 999.
Commit `3d84eb2` uses representable uint8_t value 255, still an unsupported
enumerator. Existing rejection assertions stay intact. One repair round resolved
all failures. The release workflow did not run; no tags or merges were created.
Real vLLM calibration and mutation testing remain pending/not run respectively.
