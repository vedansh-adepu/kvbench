# kvbench

A scriptable C++20 LLM inference capacity planner for KV-cache budgets and
continuous-batching workloads. It is a planning model, not an inference engine
or a hardware benchmark. Calibration evidence is pending.

## Why I built it

I needed to size KV-cache capacity before provisioning GPUs, and the math I kept
redoing by hand belonged in a tool. kvbench generalizes the capacity math behind
right-sizing max_model_len in production.

## Build and quickstart

Use CMake 3.20+, a C++20 compiler and Ninja for the presets. Dependencies are
pinned archives; see [offline builds](docs/build.md). Illustrative commands:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
./build-release/kvbench version
./build-release/kvbench simulate --config examples/workloads/llama7b_chat.json --format json
./build-release/kvbench budget --config examples/workloads/llama7b_chat.json --max-memory-gib 80 --format json
./build-release/kvbench schedule --config examples/workloads/scheduled_burst.json --format json
```

Existing examples are schema-v1 compatibility fixtures and emit deprecation
warnings. Prefer schema v2 with explicit checkpoint weights and runtime overheads.
See [configuration](docs/config-reference.md) and [migration](docs/migration.md).

## What it models

- Standard MHA/GQA/MQA, latent MLA, sliding-window and hybrid layer cache bytes.
- Packed dtypes and configurable per-tensor, per-token-head or group metadata.
- Weights (explicit or estimated) and separate runtime/activation/graph allowances.
- Whole-block budgets, phase-correct static paging and uncapped capacity math.
- Decode-first continuous batching, chunked prefill, recompute preemption and arrivals.

Use `budget`, `simulate`, `schedule`, `sweep`, `compare`, `graph`, `version` and
`completion`; see [CLI](docs/cli.md). JSON results carry `kvbench_result: 2`.
Memory fields use bytes, with GiB inputs named explicitly. Events stream to
JSONL only when requested and are never retained as a history by the scheduler.

## How it works

The [shared memory model](docs/model.md), [weights/static model](docs/static-model.md),
[budget](docs/budget.md) and [scheduler](docs/scheduler.md) document the code's
formulas and assumptions. Arrival simulation remains separate from static
worst-case output; neither is presented as measured throughput.

## Limitations

- Runtime and time-model defaults are UNCALIBRATED; no real vLLM accuracy evidence yet.
- Hybrid cache-group padding and backend-specific physical layouts are not reproduced.
- MLA weights require explicit checkpoint size; cache dimensions do not determine weights.
- No prefix sharing, speculative decoding, distributed serving or kernel execution.
- Remote CI, coverage and sanitizer/fuzz/static-analysis gates are not yet verified.
- The import/calibration commands, generated results and benchmarks are upcoming phases.

## License

MIT. See [LICENSE](LICENSE).
