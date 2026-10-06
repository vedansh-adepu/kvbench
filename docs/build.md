# Building

CMake 3.20+, a C++20 compiler and Ninja are required by the presets. Python 3
is needed for CLI/docs tests. On Windows, use a Visual Studio Developer shell
with MSVC in PATH; CI configures that environment explicitly.
Illustrative online build: `cmake --preset dev`, `cmake --build --preset dev`,
`ctest --preset dev`. Project warnings can be treated as errors; dependencies
are not given project warning flags. The package version comes from CMake and
is generated into a header.

FetchContent dependencies are pinned by tag-qualified archive URLs and SHA256:
nlohmann/json 3.12.0, CLI11 2.5.0, Catch2 3.8.1. JSON and CLI parsing use those libraries; Catch2 is fetched only when tests
are enabled.

For an offline first configure, populate the three source directories from the
verified archives, then pass `-DFETCHCONTENT_FULLY_DISCONNECTED=ON` and
`-DFETCHCONTENT_SOURCE_DIR_JSON=/path/json-3.12.0`,
`-DFETCHCONTENT_SOURCE_DIR_CLI11=/path/CLI11-2.5.0`,
`-DFETCHCONTENT_SOURCE_DIR_CATCH2=/path/Catch2-3.8.1` to `cmake --preset dev`.
Missing offline sources fail with a specific error instead of downloading.
Caller-provided source directories are trusted and bypass archive verification;
verify the pinned hashes yourself before extracting them. Builds may override
preset output with `-B /path/outside/repository`.

`asan-ubsan` enables ASan and non-recovering UBSan. `coverage` enables compiler
coverage instrumentation. `fuzz` selects the Clang/libFuzzer configuration;
it builds the bounded config/log targets documented in [quality](quality.md).
Presets are not evidence of successful CI or numerical correctness.
