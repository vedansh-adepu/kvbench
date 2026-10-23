# Published-log validation

The sample contains **14 records, 13 unique input-observation tuples,
and 6 issue clusters** for sensitivity analysis. Every reported
token count and displayed concurrency is consistent with the reconstructed
rounded-memory interval under the stated source/layout/config assumptions.
This is interval consistency, not exact capacity validation or a GPU experiment.
The centre residual has an unexplained directional imbalance; no correction
was fitted.

## Reproduction

Run `python3 scripts/reproduce_published_logs.py --check` from a checkout.
The standard-library script reads [the admitted inputs](published-log-inputs.csv),
recomputes the interval and sign tables, and compares them with this document.
Geometry is transcribed from the config snapshots cited below, identified by
SHA-256; no network or GPU is needed. `python3 tests/published_log_checks.py`
also checks boundary and failure cases. Output remains conditional on the
source/layout/config assumptions described here.

## Method and completeness

The admission rule requires a reported vLLM version, model and captured
model config, resolved runtime model dtype, KV dtype, max_model_len, tensor
parallel size, available KV memory and reported token count. Block size must
be printed or established from that version's source default for the decoder
attention backend named in the log. GPU identity is not required. A config's
declared dtype cannot substitute for runtime dtype when KV dtype is auto.

Settings belong to the same startup record; unrelated comments/runs cannot
supply missing fields. Ambiguous versions, layouts, backend/default choices or
settings remain excluded. These are selected public reports, not a random
sample of deployments. Reporters' hardware produced the original startup logs;
all analysis here ran on Apple M2 CPU, with no local serving or GPU run.

For each admitted record:

1. Preserve its issue/comment URL, author/date, message hash and dated config
   hash. Derive geometry from the text config for a uniform full-attention
   group; no automatic claim extends to hybrid, shared or padded layouts.
2. Trace the reported version's memory formatter and capacity formula. Five
   source refs print raw GiB with %.2f; five use str(round(bytes/GiB, 2)). All
   round to two decimals; none truncates. Invert the actual formatter over
   integer bytes, including binary-float ties, and check both boundary neighbours.
3. Evaluate the inclusive lower/upper byte bounds and floor of their midpoint
   through an independent Python reference and the actual kvbench CLI. The
   CPU analysis recorded 42 matching low/centre/high evaluations.
4. Compare reported tokens with the token bounds and reported concurrency with
   the bounds formatted to the original display precision. A printed GiB value
   is not a known exact pool. Record reservations and version-specific report
   semantics separately; do not tune parameters to make residuals disappear.

All admitted lengths are block aligned, with one modeled dense full-attention
group and context parallelism one. Older raw-block token reports and later
int(concurrency * max_model_len) reports agree for this restricted geometry.
The later text-only cases print memory before an optional multimodal frontend
reservation; that phase distinction remains a limitation for other workloads.
See the [capacity model](vllm-capacity-model.md) and
[compatibility contract](vllm-compat-capacity.md).

## Admitted inputs and public provenance

Short record IDs identify the archived records; each links to its original
issue or comment. Model configs are dated snapshots, not proof of the exact
historical config used by a reporter. Their captured hashes follow the tables.
Source links identify the captured version paths; public pages can change.

| Record | vLLM version | Model | Runtime dtype | KV dtype | Decoder backend | Block | Max length | TP |
|---|---|---|---|---|---|---:|---:|---:|
| [3dc928a1](https://github.com/vllm-project/vllm/issues/27364) | v0.11.1rc2.dev206+g09a7e6f61.d20251022 | Qwen/Qwen3-VL-4B-Instruct-FP8 | bfloat16 | auto | FLASH_ATTN | 16 | 4096 | 1 |
| [f6b4b71f](https://github.com/vllm-project/vllm/issues/27706) | v0.11.1rc4.dev39+g53a56e658 | Qwen/Qwen3-VL-2B-Instruct | bfloat16 | auto | TRITON_ATTN | 16 | 2048 | 1 |
| [56af9991](https://github.com/vllm-project/vllm/issues/27706#issuecomment-3505573367) | v0.11.1rc6.dev202+g67a2da890 | Qwen/Qwen3-VL-2B-Instruct | bfloat16 | auto | TRITON_ATTN | 16 | 16384 | 1 |
| [bfe01593](https://github.com/vllm-project/vllm/issues/27706#issuecomment-3506148044) | v0.11.1rc6.dev202+g67a2da890 | Qwen/Qwen3-VL-2B-Instruct | bfloat16 | auto | TRITON_ATTN | 16 | 16384 | 1 |
| [c0530a77](https://github.com/vllm-project/vllm/issues/27706#issuecomment-3515225879) | v0.11.1rc7.dev4+ga5a790eea | Qwen/Qwen3-VL-2B-Instruct | bfloat16 | auto | TRITON_ATTN | 16 | 16384 | 1 |
| [325cbbda](https://github.com/vllm-project/vllm/issues/27706#issuecomment-3706859567) | v0.14.0rc1.dev220+g0eee877f6 | Qwen/Qwen3-VL-2B-Instruct | bfloat16 | auto | TRITON_ATTN | 16 | 8192 | 1 |
| [7501f4bb](https://github.com/vllm-project/vllm/issues/32611#issuecomment-3776589979) | v0.1.dev13062+gce58b0181 | facebook/opt-125m | float16 | auto | FLASH_ATTN | 16 | 2048 | 1 |
| [de84ca03](https://github.com/vllm-project/vllm/issues/39010) | v0.19.0 | mistralai/Devstral-Small-2-24B-Instruct-2512 | bfloat16 | fp8_e4m3 | ROCM_ATTN | 16 | 128000 | 2 |
| [5963560e](https://github.com/vllm-project/vllm/issues/39010#issuecomment-4188185503) | v0.18.1 | mistralai/Devstral-Small-2-24B-Instruct-2512 | bfloat16 | fp8_e4m3 | TRITON_ATTN | 16 | 128000 | 2 |
| [e574d3aa](https://github.com/vllm-project/vllm/issues/39010#issuecomment-4188438699) | v0.19.0 | mistralai/Devstral-Small-2-24B-Instruct-2512 | bfloat16 | fp8_e4m3 | ROCM_ATTN | 16 | 128000 | 2 |
| [17094df7](https://github.com/vllm-project/vllm/issues/39010#issuecomment-4188525886) | v0.19.0 | mistralai/Devstral-Small-2-24B-Instruct-2512 | bfloat16 | fp8_e4m3 | ROCM_ATTN | 16 | 128000 | 2 |
| [8eba5533](https://github.com/vllm-project/vllm/issues/50705#issuecomment-5175386866) | v0.26.0 | Qwen/Qwen3-8B | bfloat16 | fp8 | FLASHINFER | 16 | 40960 | 1 |
| [579324ce](https://github.com/vllm-project/vllm/issues/50705#issuecomment-5175386866) | v0.26.0 | Qwen/Qwen3-8B | bfloat16 | auto | FLASH_ATTN | 16 | 40960 | 1 |
| [22d8422f](https://github.com/vllm-project/vllm/issues/52802) | v0.27.1 | Qwen/Qwen3-0.6B | bfloat16 | auto | FLASH_ATTN | 16 | 40960 | 1 |

## Interval table

Bounds are inclusive. Concurrency bounds below use the source display precision,
not exact unrounded concurrency. Delta is centre-predicted tokens minus reported
tokens; the centre is a diagnostic point inside a memory range, not a measured
pool. The repeated Devstral record stays visible but is excluded from statistics.

| Record | Printed GiB | Reported tokens | Token interval | Centre delta | Reported concurrency | Displayed concurrency interval | Duplicate of |
|---|---:|---:|---|---:|---:|---|---|
| [3dc928a1](https://github.com/vllm-project/vllm/issues/27364) | 102.74 | 748,144 | [748,080, 748,160] | -16 | 182.65 | [182.64, 182.66] | none |
| [f6b4b71f](https://github.com/vllm-project/vllm/issues/27706) | 10.12 | 94,784 | [94,688, 94,784] | -48 | 46.28 | [46.23, 46.28] | none |
| [56af9991](https://github.com/vllm-project/vllm/issues/27706#issuecomment-3505573367) | 5.05 | 47,312 | [47,232, 47,312] | -48 | 2.89 | [2.88, 2.89] | none |
| [bfe01593](https://github.com/vllm-project/vllm/issues/27706#issuecomment-3506148044) | 5.85 | 54,784 | [54,720, 54,816] | -16 | 3.34 | [3.34, 3.35] | none |
| [c0530a77](https://github.com/vllm-project/vllm/issues/27706#issuecomment-3515225879) | 5.85 | 54,784 | [54,720, 54,816] | -16 | 3.34 | [3.34, 3.35] | none |
| [325cbbda](https://github.com/vllm-project/vllm/issues/27706#issuecomment-3706859567) | 6.58 | 61,648 | [61,552, 61,648] | -48 | 7.53 | [7.51, 7.53] | none |
| [7501f4bb](https://github.com/vllm-project/vllm/issues/32611#issuecomment-3776589979) | 18.0 | 524,416 | [524,128, 524,432] | -128 | 256.06 | [255.92, 256.07] | none |
| [de84ca03](https://github.com/vllm-project/vllm/issues/39010) | 11.87 | 311,264 | [311,024, 311,280] | -112 | 2.43 | [2.43, 2.43] | none |
| [5963560e](https://github.com/vllm-project/vllm/issues/39010#issuecomment-4188185503) | 11.84 | 310,352 | [310,240, 310,496] | +16 | 2.42 | [2.42, 2.43] | none |
| [e574d3aa](https://github.com/vllm-project/vllm/issues/39010#issuecomment-4188438699) | 11.87 | 311,264 | [311,024, 311,280] | -112 | 2.43 | [2.43, 2.43] | de84ca03 |
| [17094df7](https://github.com/vllm-project/vllm/issues/39010#issuecomment-4188525886) | 11.94 | 313,056 | [312,864, 313,120] | -64 | 2.45 | [2.44, 2.45] | none |
| [8eba5533](https://github.com/vllm-project/vllm/issues/50705#issuecomment-5175386866) | 12.2 | 177,728 | [177,600, 177,744] | -64 | 4.34 | [4.34, 4.34] | none |
| [579324ce](https://github.com/vllm-project/vllm/issues/50705#issuecomment-5175386866) | 12.2 | 88,864 | [88,800, 88,864] | -32 | 2.17 | [2.17, 2.17] | none |
| [22d8422f](https://github.com/vllm-project/vllm/issues/52802) | 13.86 | 129,744 | [129,712, 129,808] | +16 | 3.17 | [3.17, 3.17] | none |

## Residual signs and independence

Deduplicate identical version/config/dtype/backend/block/length/TP/memory/token/
concurrency tuples, retaining the duplicate's provenance. The exact two-sided
sign test excludes zero residuals and doubles the smaller binomial tail under
equiprobable independent signs, capped at one.

| Analysis unit | Negative | Zero | Positive | Exact two-sided p |
|---|---:|---:|---:|---|
| Unique tuple | 11 | 0 | 2 | 23/1024 = 0.0224609375 |
| One mean per issue | 5 | 0 | 1 | 7/32 = 0.21875 |
| One median per issue | 5 | 0 | 1 | 7/32 = 0.21875 |

The tuple p-value is nominal: multiple reports within an issue do not establish
independent trials. At the issue level, the same directional pattern has much
weaker evidence. The 6 issues are proposed independent units, but independence
even across issues is unproven. These are declared sensitivity analyses, not
permission to choose whichever p-value is preferred. The issue-level result
does not prove absence of bias, and neither test supports a population accuracy
claim for this selected sample.

| Issue | Records | Unique tuples | Exact mean delta | Median delta |
|---|---:|---:|---:|---:|
| [27364](https://github.com/vllm-project/vllm/issues/27364) | 1 | 1 | -16 | -16 |
| [27706](https://github.com/vllm-project/vllm/issues/27706) | 5 | 5 | -176/5 | -48 |
| [32611](https://github.com/vllm-project/vllm/issues/32611) | 1 | 1 | -128 | -128 |
| [39010](https://github.com/vllm-project/vllm/issues/39010) | 4 | 3 | -160/3 | -64 |
| [50705](https://github.com/vllm-project/vllm/issues/50705) | 2 | 2 | -48 | -48.0 |
| [52802](https://github.com/vllm-project/vllm/issues/52802) | 1 | 1 | 16 | 16 |

One unique upper tie endpoint, appearing in two duplicate records, was excluded
by the corrected formatter preimage. No point residual or interval-membership
result changed. The negative imbalance remains unexplained. Rounded-memory
phase, historical config/layout differences and correlated sampling need direct
observations; a reservation alone is not asserted to explain negative residuals.

## Source rules and configuration snapshots

| Reported version | Memory formatter | Memory path | Capacity path |
|---|---|---|---|
| v0.1.dev13062+gce58b0181 | round_2_str | [worker](https://github.com/vllm-project/vllm/blob/ce58b0181/vllm/v1/worker/gpu_worker.py#L356-L359), [conversion](https://github.com/vllm-project/vllm/blob/ce58b0181/vllm/utils/mem_utils.py#L23-L24) | [formula](https://github.com/vllm-project/vllm/blob/ce58b0181/vllm/v1/core/kv_cache_utils.py#L1284-L1313) |
| v0.11.1rc2.dev206+g09a7e6f61.d20251022 | printf_2f | [worker](https://github.com/vllm-project/vllm/blob/09a7e6f61/vllm/v1/worker/gpu_worker.py#L317-L320), [conversion](https://github.com/vllm-project/vllm/blob/09a7e6f61/vllm/v1/worker/gpu_worker.py#L251) | [formula](https://github.com/vllm-project/vllm/blob/09a7e6f61/vllm/v1/core/kv_cache_utils.py#L1178-L1207) |
| v0.11.1rc4.dev39+g53a56e658 | printf_2f | [worker](https://github.com/vllm-project/vllm/blob/53a56e658/vllm/v1/worker/gpu_worker.py#L320-L323), [conversion](https://github.com/vllm-project/vllm/blob/53a56e658/vllm/v1/worker/gpu_worker.py#L254) | [formula](https://github.com/vllm-project/vllm/blob/53a56e658/vllm/v1/core/kv_cache_utils.py#L1206-L1235) |
| v0.11.1rc6.dev202+g67a2da890 | printf_2f | [worker](https://github.com/vllm-project/vllm/blob/67a2da890/vllm/v1/worker/gpu_worker.py#L350-L353), [conversion](https://github.com/vllm-project/vllm/blob/67a2da890/vllm/v1/worker/gpu_worker.py#L284) | [formula](https://github.com/vllm-project/vllm/blob/67a2da890/vllm/v1/core/kv_cache_utils.py#L1206-L1235) |
| v0.11.1rc7.dev4+ga5a790eea | printf_2f | [worker](https://github.com/vllm-project/vllm/blob/a5a790eea/vllm/v1/worker/gpu_worker.py#L353-L356), [conversion](https://github.com/vllm-project/vllm/blob/a5a790eea/vllm/v1/worker/gpu_worker.py#L287) | [formula](https://github.com/vllm-project/vllm/blob/a5a790eea/vllm/v1/core/kv_cache_utils.py#L1206-L1235) |
| v0.14.0rc1.dev220+g0eee877f6 | printf_2f | [worker](https://github.com/vllm-project/vllm/blob/0eee877f6/vllm/v1/worker/gpu_worker.py#L362-L365), [conversion](https://github.com/vllm-project/vllm/blob/0eee877f6/vllm/v1/worker/gpu_worker.py#L296) | [formula](https://github.com/vllm-project/vllm/blob/0eee877f6/vllm/v1/core/kv_cache_utils.py#L1282-L1311) |
| v0.18.1 | round_2_str | [worker](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/v1/worker/gpu_worker.py#L457-L460), [conversion](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/utils/mem_utils.py#L23-L24) | [formula](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/v1/core/kv_cache_utils.py#L1293-L1322) |
| v0.19.0 | round_2_str | [worker](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/worker/gpu_worker.py#L437-L440), [conversion](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/utils/mem_utils.py#L23-L24) | [formula](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/core/kv_cache_utils.py#L1296-L1325) |
| v0.26.0 | round_2_str | [worker](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/worker/gpu_worker.py#L561-L564), [conversion](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/utils/mem_utils.py#L30-L31) | [formula](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/kv_cache_utils.py#L1819-L1829) |
| v0.27.1 | round_2_str | [worker](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/gpu_worker.py#L564-L567), [conversion](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/utils/mem_utils.py#L30-L31) | [formula](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_utils.py#L1877-L1887) |

| Model config snapshot | Captured date | SHA-256 |
|---|---|---|
| [Qwen/Qwen3-VL-4B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8/resolve/main/config.json) | 2026-10-18 | `48bc97743d741920de3455f7a6f0a807b0c3ae692e662a0864f188ca528f92b2` |
| [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/config.json) | 2026-10-18 | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| [facebook/opt-125m](https://huggingface.co/facebook/opt-125m/resolve/main/config.json) | 2026-10-18 | `54162c9cd0377110df1859dfabebb106fc0988c8aadb88207ce1ee0d34985df6` |
| [mistralai/Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/resolve/main/config.json) | 2026-10-18 | `8d570ae877badd9ff5442afdc726385aba07f5757ba875a7491bde9549457e72` |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json) | 2026-10-18 | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json) | 2026-10-18 | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |

## What this establishes and leaves open

The sample supports conditional interval consistency with version-correct
rounding and modeled geometry. It does not identify exact effective pool bytes,
validate startup profiling from configuration, explain the centre residual sign,
prove general capacity accuracy, or measure latency, throughput or GPU behavior.

The planned exact-byte GPU validation supplies explicit kv_cache_memory_bytes
at predeclared block boundaries, pins model/config and engine revisions, and
dumps effective bytes, resolved page/group geometry, dtype, blocks and reported
capacity from the same real startup. It also observes pre/post-reservation
memory to distinguish arithmetic from display precision and memory phases.
That exact test is still pending; this published-log analysis is not its result.
No offset, overhead, geometry correction or fitted bias adjustment was applied.
