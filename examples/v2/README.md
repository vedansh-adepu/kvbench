# Planning examples, not hardware measurements

All configs use explicit schema-v2 units. Standard models use architecture-based
weight estimates and UNCALIBRATED runtime allowances; checkpoint measurements
must replace those estimates for real provisioning. The 24 GB-class example
uses exactly 24,000,000,000 bytes (decimal GB), not 24 GiB.

`llama3-8b-like` and `long-context` contrast chat/long contexts. The Qwen-like
full model does not fit a single 80 GiB GPU under its estimated resident weights.
The TP4 approximation uses one 80 GiB pool per rank, divides KV heads by four
and query heads by four, and divides full estimated weights by four, rounding up bytes. Runtime
allowances are not divided. Real TP can replicate tensors/pad partitions: this
is a per-rank planning approximation, not measured checkpoint sharding. Tests
check the mapping against the full-model tool result.

The DeepSeek-like example is explicitly cache-only: weights and runtime
allowances are zero to isolate latent KV geometry. It does NOT claim that the
full checkpoint fits a 4 GiB GPU. Supply real per-rank weights before provisioning.
The hybrid example is a toy, not a claim of a particular checkpoint's layout.

Bursty/smoothed files contain identical requests and differ only in the second
arrival. Their isolated GQA cache pool has 40 blocks; weights/runtime allowances
are excluded to isolate scheduling behavior. Times are UNCALIBRATED. Run
`schedule`, not static `simulate`, to compare arrivals.

Apply this mapping only when the KV head count is divisible by TP. Engines may
replicate KV heads for MQA or TP larger than the KV head count; that case is not
represented by simply dividing heads.
