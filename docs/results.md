# Generated planning results

Model output, not a hardware measurement; time model uncalibrated unless stated.

Regenerate with `python3 scripts/gen_results.py --binary build-release/kvbench`.

## Architecture cache geometry

| Geometry | Data bytes/token |
| --- | ---: |
| MHA (32 KV heads) | 524,288 |
| GQA (8 KV heads) | 131,072 |
| MLA latent (61 layers, 512+64) | 70,272 |

MLA example excludes weights/runtime solely to isolate cache geometry.

## 24 GB-class budget at different maximum lengths

Assumed total GPU capacity is exactly 24,000,000,000 bytes, utilization=0.9;
weights are estimated, runtime allowances are UNCALIBRATED. Maximum length
includes all cached tokens, so decode_tokens=0 for this capacity comparison.

| Cached length | Blocks/sequence | Pool blocks | Max sequences |
| ---: | ---: | ---: | ---: |
| 8192 | 512 | 1745 | 3 |
| 4096 | 256 | 1745 | 6 |
| 2048 | 128 | 1745 | 13 |

## KV dtype capacity with explicit metadata assumptions

Same estimated bf16 weights/runtime budget, cached length=4096. These
metadata widths are planning assumptions; verify your exact backend/version.

| KV dtype / metadata | Block data+metadata bytes | Constant metadata bytes | Pool blocks | Max sequences |
| --- | ---: | ---: | ---: | ---: |
| fp16 / none, scale=4, zp=0 | 2,097,152 | 0 | 1745 | 6 |
| fp8_e4m3 / per_tensor, scale=4, zp=0 | 1,048,576 | 256 | 3490 | 13 |
| int8 / per_token_head, scale=4, zp=0 | 1,081,344 | 0 | 3385 | 13 |
| int4 / per_token_head, scale=4, zp=2 | 573,440 | 0 | 6383 | 24 |

## Arrival sensitivity

Isolated 40-block GQA fixture, identical requests. Weights/runtime are
excluded; base=1 ms, prefill=0.01 ms/token, decode=0.1 ms/sequence.

| Arrival pattern | Peak blocks | Preemptions | Recomputed tokens | p99 TTFT (modeled ms) | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Burst | 40 | 3 | 256 | 92.620 | completed |
| Smoothed | 36 | 0 | 0 | 10.220 | completed |

## Independent sweep CSV excerpt

The following is actual CLI CSV, not a hand-maintained approximation.

```csv
context_tokens,concurrent_requests,peak_bytes,fits,risk,max_num_batched_tokens
2048,1,1409286144,true,ok,2048
2048,2,1744830464,true,ok,2048
2049,1,1411383296,true,ok,2048
2049,2,1749024768,true,ok,2048
```
