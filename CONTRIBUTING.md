# Contributing

Build with CMake 3.20+, a C++20 compiler and Ninja. `cmake --preset dev`,
`cmake --build --preset dev`, `ctest --preset dev` run the local suite.
For disconnected builds see [build documentation](docs/build.md).

Keep commits focused and honest (`fix(scope): summary`, `test(scope): summary`).
Do not rewrite history or commit build outputs, private logs or generated fuzz
corpora. Add exact/generative tests for semantic changes; run the full suite before
committing. Update public unit/guarantee comments and formulas alongside code.

Use `clang-format -i src/*.cpp include/kvbench/*.hpp tests/*.cpp fuzz/*.cpp` and
`python3 scripts/analyze.py --build build-dev` with clang-tidy/cppcheck installed.
Run the coverage/sanitizer/fuzz checks in [quality](docs/quality.md), reporting
unavailable tools/runtime failures honestly. Project warnings are errors in presets.

Regenerate deterministic results with
`python3 scripts/gen_results.py --binary build-release/kvbench`; CTest verifies
exact equality. CPU benchmarks use `python3 bench/run.py --binary build-release/kvbench`;
their output is a measurement of the planner, not the GPU inference engine.
Never present a synthetic/public-unknown-config log as real calibration evidence.
