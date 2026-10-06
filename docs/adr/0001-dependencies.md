# 0001 — Pinned parsing, CLI and testing libraries

Use nlohmann/json 3.12.0, CLI11 2.5.0 and Catch2 3.8.1 through hash-verified
FetchContent archives. Reusing strict parsing/CLI libraries removes handwritten
numeric/Unicode handling. Duplicate keys/depth limits remain explicit policy.
Trade-off: first online configuration needs downloads; offline callers must
provide verified source directories. Catch2 is fetched only for test builds.
