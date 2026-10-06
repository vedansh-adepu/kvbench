# Migrating v1 configs

A document without `schema_version` is v1. The planner loader returns a
schema-v2 config and a deprecation warning; unknown v1 fields produce warnings
and are ignored. V2 unknown fields are errors. Warnings are returned by the
library; the CLI prints them to stderr. Prototype Python/C++ types and coupled
batch sweep flags are removed; v1 file compatibility remains.

| v1 field | v2 mapping |
| --- | --- |
| model.layers / attention_heads / kv_heads | num_layers / num_attention_heads / num_kv_heads |
| model.head_dim / hidden_size | Same names; safe derivation if missing |
| model.dtype | Used for v1 activation allowance and default KV dtype |
| system.gpu_memory_gb | hardware.total_gpu_gib; v1's GB spelling meant GiB |
| system.reserved_memory_gb | engine.non_torch_overhead_gib |
| system.kv_quantization | engine.kv_dtype; idealized no-metadata compatibility |
| system.page_size_tokens | engine.block_size |
| workload.concurrent_requests | Same static field and engine.max_num_seqs |
| workload.batch_size×prefill_chunk_size | engine.max_num_batched_tokens |
| workload.context_tokens / decode_tokens | Same static fields |
| root requests[] | workload.requests[]; arrivals do not overwrite static fields |

For v1, utilization is 1, explicit weights are 0 and CUDA graph allowance is 0:
this preserves the old requirement that reserved memory includes weights.
The previous prefill scratch formula maps to a fixed, UNCALIBRATED activation
allowance, rounded up to bytes: layers × hidden_size × min(context,chunk) ×
min(batch,concurrency) × model dtype bits / 32. Set measured explicit v2 values
instead; the compatibility mapping is not an engine-accuracy claim.

Prefer explicit checkpoint weights and independently measured runtime overheads
in v2. Do not carry the old reserved amount over and also subtract weights a
second time. Use [CLI reference](cli.md) for commands, versioned output and exit codes.
