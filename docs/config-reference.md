# Configuration schema v2

`parse_planner_config` accepts `schema_version: 2` and returns a validated
`PlannerConfig` plus warnings. Schema-less input is v1; see [migration](migration.md).
All CLI planning commands use this schema and the same validated memory model.

Only listed fields are accepted in v2. Sections must be objects, booleans must
be JSON booleans, counts must be JSON integers (even `8.0` is rejected), and
numbers must be finite. Duplicate keys, invalid UTF-8/surrogates, inputs above
16 MiB and depth above 128 are rejected. No implicit string-to-number conversion.
Memory alternatives `_bytes` and `_gib` are mutually exclusive. Bytes are exact
nonnegative uint64; GiB values round down to bytes and must remain below 2^64.
All values below are planning defaults, not calibration evidence.

## Model

| Field | Unit/default | Range/meaning |
| --- | --- | --- |
| name | string, unnamed-model | Output label |
| num_layers | required count | 1–4096 |
| num_attention_heads | required count | 1–65536 |
| num_kv_heads | defaults to attention heads | 1–65536; non-divisibility warns |
| head_dim | derived from hidden_size/heads if absent | 1–1048576; exact division required |
| hidden_size | derived from heads×head_dim if absent | 1–16777216 |
| attention_type | standard | standard/full, sliding, mla |
| kv_lora_rank | 0 | MLA requires 1–1048576 |
| qk_rope_head_dim | 0 | MLA requires 1–1048576 |
| sliding_window | 0 tokens | Positive for uniform sliding layers |
| layer_types | absent, uniform model | One string or `{type,window_tokens}` per layer; exact layer count |

## Weights

| Field | Unit/default | Range/meaning |
| --- | --- | --- |
| weights_bytes / weights_gib | absent | Explicit size preferred; otherwise estimator inputs below |
| dtype | bf16 | fp32/fp16/bf16/fp8 (alias e4m3)/fp8_e4m3/fp8_e5m2/int8/int4 |
| intermediate_size | hidden_size×4 | 1–16777216; architecture estimate assumption |
| vocab_size | 32000 | 1–16777216 |
| tie_word_embeddings | true | Boolean |
| gated_mlp | true | Boolean; three matrices instead of two |
| num_experts | 1 | 1–65536, all resident |
| num_shared_experts | 0 | 0–65536 |
| quantization_overhead_bytes | 0 bytes | Explicit uint64 checkpoint metadata allowance |

The [weight estimator](static-model.md) uses these fields; explicit weights are preferred.

## Hardware and engine

| Field | Unit/default | Range/meaning |
| --- | --- | --- |
| hardware.total_gpu_bytes / total_gpu_gib | 24 GiB | Positive; single GPU |
| engine.gpu_memory_utilization | 0.9 | Finite (0,1] |
| non_torch_overhead_bytes / non_torch_overhead_gib | 0.25 GiB | Nonnegative bytes; UNCALIBRATED |
| activation_peak_bytes / activation_peak_gib | 1 GiB | Nonnegative bytes; UNCALIBRATED fixed allowance |
| cudagraph_memory_bytes / cudagraph_memory_gib | 0.5 GiB | Nonnegative bytes; UNCALIBRATED |
| block_size | 16 tokens | 1–1048576 |
| swa_extra_blocks | 1 | 0–1048576; see [model](model.md) |
| kv_dtype | fp16 | Exact spellings in [quantization](quantization.md) |
| scale_mode | none | none/per_tensor/per_token_head/group |
| scale_bytes | 4 bytes | Quantized modes require 1–16 |
| zero_point_bytes | 0 bytes | Quantized modes allow 0–16 |
| group_size | 64 elements | Group mode requires 1–16777216 |
| max_num_seqs | 256 | 1–1000000 |
| max_num_batched_tokens | 2048 | 1–1000000000 |
| enable_chunked_prefill | true | Boolean |
| num_blocks | absent, derive from budget | Nonnegative uint64 override; budget feasibility checked in budget phase |
| preemption_mode | recompute | Only recompute supported |
| base_step_ms | 1 ms | Finite nonnegative |
| prefill_ms_per_token | 0.01 ms/token | Finite nonnegative |
| decode_ms_per_seq | 0.1 ms/sequence | Finite nonnegative |
| max_simulated_time_ms | 86400000 ms | Finite positive; explicit truncation status in scheduler phase |
| max_steps | 1000000 | Positive uint64 |
| calibrated | false | User-supplied provenance flag; not a proof of accuracy |

The three step-time coefficients cannot all be zero. Cache geometry and engine
block settings must remain consistent when callers mutate a public config.

## Workload

| Field | Unit/default | Range/meaning |
| --- | --- | --- |
| context_tokens | 1 token | Positive uint64 |
| decode_tokens | 0 tokens | Nonnegative; context+decode must fit uint64 |
| concurrent_requests | 1 | Positive uint64; calculation overflow is rejected |
| requests | absent | At most 1000000 request objects |
| requests[].id | request-N | Nonempty unique string; N is zero-based input order |
| requests[].arrival_ms | 0 ms | Finite nonnegative; fractional milliseconds preserved |
| requests[].input_tokens | required positive | Positive uint64 |
| requests[].output_tokens | 0 tokens | Nonnegative; input+output must fit uint64 |

Requests remain independent from static worst-case fields. Input order breaks
ties. Later schedule output must distinguish modeled time from measurements.
