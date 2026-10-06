# Quantization assumptions

Cache storage and metadata arithmetic are tested independently of kernel
availability. Supported planning dtypes are fp32, fp16, bf16, fp8_e4m3,
fp8_e5m2, int8 and int4. See [model](model.md) for packing and scale formulas.
Default metadata parameters are 4 scale bytes, 0 zero-point bytes and 64
elements per group. They are not verified engine defaults: choose a scale
mode and the correct metadata widths for your backend.

| Planning mode | Engine evidence and boundary |
| --- | --- |
| FP8 per-tensor | Historical vLLM paths use per-tensor scales. |
| int8/int4 per-token-head | Listed in vLLM v0.31.0's KVQuantMode; availability is backend dependent. |
| Group scales | Generic configurable arithmetic, not a claim that any particular vLLM kernel uses it. |
| NVFP4 | Listed by vLLM; not a kvbench dtype yet. Do not substitute int4 as an exact NVFP4 layout. |

[Versioned vLLM source documentation](https://docs.vllm.ai/en/v0.31.0/api/vllm/v1/kv_cache_interface/#vllm.v1.kv_cache_interface.KVQuantMode)
lists FP8 per-tensor, per-token-head modes and packed NVFP4. Check your exact
engine version, device and backend. int4 can require asymmetric zero points;
configure their bytes instead of assuming only nibble data exists. Numeric
capacity calculations do not establish execution support or model quality.
