# kvbench

A fast, scriptable C++20 LLM inference capacity planner for KV-cache budgets and
continuous-batching workloads.

Many web calculators compute the KV-cache formula. kvbench applies vLLM-style
block budgeting with weights and runtime overheads, simulates continuous batching
with block admission, chunked prefill and recompute preemption, and compares its
predictions with a supplied vLLM startup log. Arrival patterns change pool
occupancy and preemptions. It is a planning model, not an inference engine or a
hardware benchmark. **Calibration evidence is pending.**

## Why I built it

I needed to size KV-cache capacity before provisioning GPUs, and the math I kept
redoing by hand belonged in a tool. kvbench generalizes the capacity math behind
right-sizing max_model_len in production.

## Quickstart

Install CMake 3.20+, a C++20 compiler, Ninja and Python 3 for tests. Build steps
are illustrative and depend on your compiler environment:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

The following commands are executed by the documentation test; replace the
binary path with your build location when needed.

<!-- tested: quickstart -->
```sh
./build-release/kvbench version
./build-release/kvbench simulate --config examples/v2/llama3-8b-like.json --format json
./build-release/kvbench budget --config examples/v2/llama3-8b-like.json --format json
./build-release/kvbench schedule --config examples/v2/bursty-arrivals.json --format json
```

For offline configuration see [build](docs/build.md). Schema-less v1 files remain
readable with deprecation warnings; see [migration](docs/migration.md).

## What it models

- MHA/GQA/MQA, latent MLA and per-layer sliding/hybrid cache geometry.
- Packed dtypes plus configurable scale/zero-point metadata.
- Explicit or estimated resident weights and separate runtime/activation/graph terms.
- Phase-correct static paging, whole-block budgets and uncapped capacity math.
- Decode-first continuous batching, FCFS chunked prefill and recompute preemption.

`budget`, `simulate`, `schedule`, `sweep`, `compare`, `import-hf`, `calibrate`,
`graph`, `completion` and `version` are documented in [CLI](docs/cli.md).
JSON results carry `kvbench_result: 2`; bytes and GiB inputs are named explicitly.
The sweep varies context and concurrency independently. Events stream to JSONL
only when requested and are not retained as a scheduler history.

## Example output

This excerpt is captured from the current tool and checked by the documentation
test. Model output, not a hardware measurement; the time model is UNCALIBRATED.

<!-- tested: arrival-output -->
```text
pattern,peak_blocks,preemptions,recomputed_tokens,p99_ttft_ms,status
burst,40,3,256,92.620,completed
smoothed,36,0,0,10.220,completed
```

The identical requests differ only in arrival timing. The example isolates a
40-block GQA pool; weights/runtime are explicitly excluded for that scheduling
fixture. See [generated results](docs/results.md) for weight-inclusive capacity,
quantization metadata, architecture comparisons and an actual sweep excerpt.

## How it works

Standard KV data = layers × KV heads × head dimension × 2 × dtype bytes.
Latent MLA uses layers × (kv_lora_rank + qk_rope_head_dim) × dtype bytes.
Checked integer math adds metadata and page rounding at the allocation boundary.

```text
JSON/HF config -> strict validation -> shared cache geometry
                                      |
                   weights + runtime budget -> whole block pool
                                      |               |
                              static worst case   arrival scheduler
                                      |               |
                             versioned results / escaped formats
                                      |
                         supplied startup log -> comparison
```

The [model](docs/model.md), [weights/static model](docs/static-model.md),
[budget](docs/budget.md) and [scheduler](docs/scheduler.md) explain the formulas.
Scheduler peaks represent active pool occupancy; a serving engine may preallocate
its whole KV pool. These outputs are not measured GPU memory or throughput.

## Calibration

Use a local HF config to import architecture, then supply checkpoint weights and
runtime inputs from your actual engine. `calibrate` compares observed startup
memory/block/token terms with predictions; missing terms remain missing.
See [calibration](docs/calibration.md). The public discussion fixture tests parsing
only, because its exact model/config is unknown. No accuracy claim is made.

## How it compares

Web tools such as [llm-mem-calculator](https://github.com/elinx/llm-mem-calculator),
[SelfHostLLM](https://github.com/erans/selfhostllm) and
[kvanta](https://kvanta.vcerny.cz) cover memory estimates and related model/config
workflows. kvbench focuses on a scriptable C++ CLI, explicit block budgets,
continuous-batching/preemption simulation, startup-log comparisons and CI-friendly
exit codes. This is a different interface and scope, not a claim of unique
formulas or better engine accuracy.

## What it is not

It does not load checkpoints, execute kernels, contact model APIs or allocate GPU
cache tensors. No API keys are needed. Timing coefficients are model inputs.

## Limitations

- Runtime/time defaults are UNCALIBRATED; real vLLM calibration evidence is pending.
- Hybrid group padding and chunked sliding-window backend layouts are approximations.
- MLA weights require explicit size; its cache dimensions do not determine projections.
- TP division examples omit replicated/padded tensors; verify real per-rank memory.
- Prefix sharing, speculative decoding and distributed execution are not modeled.
- Local macOS ASan startup is blocked; Linux sanitizers and remote CI passed ([results](docs/quality.md)).
- Bounded fuzzing and line coverage are evidence of tests, not an exhaustive proof.

## Roadmap

Collect versioned real-engine calibration fixtures, refine backend-specific hybrid
allocation and MLA projection metadata, and evaluate prefix-sharing models.

## License

MIT. See [LICENSE](LICENSE).

[Configuration](docs/config-reference.md) · [Quality](docs/quality.md) ·
[CPU performance](docs/performance.md) · [ADRs](docs/adr/0001-dependencies.md) ·
[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)
