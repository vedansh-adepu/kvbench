# Opt-in vLLM capacity reporting

Set `engine.vllm_compat` to true and supply `engine.max_model_len`. The default
mode and output schema remain unchanged. The compatibility mode uses vLLM's
double multiplication followed by ceil for the utilization budget, and reports
`C = num_blocks / ceil(max_model_len / block_size)` and `int(C * max_model_len)`.
`vllm_capacity.max_concurrency` is the fractional ratio; `log_lines` is an array
with comma-grouped tokens and two-decimal concurrency, as in the pinned
[vLLM capacity specification](vllm-capacity-model.md).

`engine.vllm_report_style` selects `combined` (default, one array element) or
`split` (two elements, for older versions). Select the style explicitly when
reproducing an older report; no version is inferred from model configuration.
The combined message first appears in stable release
[v0.28.0, lines 1869-1875](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/core/kv_cache_utils.py#L1869-L1875).
The preceding stable release
[v0.27.1, lines 2235-2240](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_utils.py#L2235-L2240)
has two logging calls. Pinned
[v0.32.0, lines 2327-2334](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/kv_cache_utils.py#L2327-L2334)
uses the combined form:

```text
GPU KV cache size: <tokens with commas> tokens, Maximum concurrency for <max_model_len with commas> tokens per request: <C to 2 decimals>x
```

The split form is:

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
