# Opt-in vLLM capacity reporting

Set `engine.vllm_compat` to true and supply `engine.max_model_len`. The default
mode and output schema remain unchanged. The compatibility mode uses vLLM's
double multiplication followed by ceil for the utilization budget, and reports
`C = num_blocks / ceil(max_model_len / block_size)` and `int(C * max_model_len)`.
`vllm_capacity.max_concurrency` is the fractional ratio; `log_lines` is an array
of the two separate vLLM strings, with comma-grouped tokens and two-decimal
concurrency, as in the pinned
[vLLM capacity specification](vllm-capacity-model.md).

```text
GPU KV cache size: <tokens with commas> tokens
Maximum concurrency for <max_model_len with commas> tokens per request: <C to 2 decimals>x
```

For exact Stage 2/3 inputs, set `engine.available_kv_pool_bytes` to a nonnegative
integer effective pool size. This bypasses utilization, weights and overhead
arithmetic; their budget fields are zero because those terms were not used.
Cache geometry must be resolved for one uniform full-attention group without
additional alignment/scale metadata, host pools, shared layers or reservations.
Unsupported layouts, integer overflow and report counts beyond the exact
double-integer range are errors. This mode does not infer
profiled memory or validate a real serving engine.

`engine.tensor_parallel_size` defaults to one. Global KV heads are divided by
TP or replicated to at least one local KV head, with divisibility validated.
For estimated Stage 1 with TP, provide explicit per-rank resident weights;
automatic distributed weight estimation is unsupported. Model dtype and KV
dtype must already be resolved. Workload concurrency and command-line bounds
remain whole-sequence planning outputs; they do not cap vLLM's report ratio.

Example engine fields (retain the rest of a valid schema-v2 config):

```json
{"vllm_compat":true,"max_model_len":40960,"tensor_parallel_size":1,
 "available_kv_pool_bytes":14882061680,"block_size":16,"kv_dtype":"bf16"}
```

The mode supports budget, compare and calibrate; simulation/scheduling are rejected.
Run `kvbench budget --config <config.json> --format json`. A valid no-fit
workload can return exit 1 while still reporting capacity. Rounded GiB logs
support interval consistency only: evaluate effective bytes at both endpoints
of the printed memory's +/-0.005 GiB interval, not an exact-byte accuracy claim.
The three log-based unit scenarios explicitly retain their assumed geometry.
