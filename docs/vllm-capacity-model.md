# vLLM KV-cache capacity model

## Pinned version and scope

This specification follows the V1 engine in vLLM **v0.32.0**, commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`. It describes reported startup
capacity, not measured serving throughput. The source links below are immutable.

Abbreviations used in citations:

- [K: core/kv_cache_utils.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py).
- [I: kv_cache_interface.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py).
- [W: worker/gpu_worker.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_worker.py).

The simple reduction below assumes one uniform full-attention group, no layer
sharing, ordinary dense/FP8 storage, and PP=PCP=DCP=1. General group equations
are separate; model architecture alone does not resolve every backend layout.

## Stage 1: available KV memory

Let `T` be device total bytes, `u` the utilization fraction, `F0` startup free
bytes, `F1` free bytes after profiling, `P1` post-profile Torch peak bytes,
`A1` post-profile Torch allocated bytes, and `G` applied CUDA-graph estimate.
For ordinary in-process weights, the source-derived budget is:

```text
requested = ceil(T * u)
non_kv = (F0 - F1) + (P1 - A1)
available_before_frontend_reserve = requested - non_kv - G
```

`request_memory` uses **ceil**, not integer truncation:
[worker/utils.py:716-726](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/utils.py#L716-L726).
The profiling equations avoid double-counting persistent Torch allocations:
[utils/mem_utils.py:303-321](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/utils/mem_utils.py#L303-L321).
The subtraction is [W:734-744](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_worker.py#L734-L744).

| Term/path | Definition and origin | Config-derivable or measured? |
|---|---|---|
| `T`, `F0`, `F1` | Device snapshots; free/allocated/non-Torch accounting in [mem_utils.py:144-160](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/utils/mem_utils.py#L144-L160) | Runtime device measurements; advertised GPU capacity is not the exact free budget. |
| `u` | `gpu_memory_utilization`, consumed by request_memory above | Launch/config input. |
| Resident weights | `model_memory_usage` supplied to profiling, [W:679-682](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_worker.py#L679-L682) | Loaded runtime footprint; a checkpoint/config estimate is not the measurement. It is already inside total persistent consumption. |
| Non-Torch allocations | Snapshot difference, `non_torch_increase`, [mem_utils.py:303-306](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/utils/mem_utils.py#L303-L306) | Runtime; includes effects outside the Torch allocator. Do not subtract it again on top of the final `non_kv` term. |
| Activation headroom | `P1-A1`, above | Runtime dummy-profile result, not determined by model parameter count. |
| CUDA graphs | Profiled when supported/enabled, applied only through `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, [W:695-711](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_worker.py#L695-L711) | Gate is config/environment; bytes are runtime-profiled. The environment default is enabled: [envs.py:2221-2223](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/envs.py#L2221-L2223). |
| External weights | request_memory subtracts externally held weights from requested engine budget, [worker/utils.py:728-759](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/utils.py#L728-L759) | Requires actual external ownership/footprint. |
| Explicit KV bytes | `kv_cache_memory_bytes` bypasses utilization-based sizing but still runs profile compilation, [W:649-674](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_worker.py#L649-L674) | Config input; not evidence that an arbitrary requested pool fits. |
| Frontend multimodal reservation | Raw-frame budget plus per-process decoder/context reservation; subtract from available, [gpu_ipc_memory.py:202-236](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/multimodal/gpu_ipc_memory.py#L202-L236) | Configuration/backend-dependent; decoder/context constants are implementation assumptions. |

The informational `Available KV cache memory` line is emitted **before** the
frontend reservation: [W:759-762,803-807](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_worker.py#L759-L807).
Its GiB value is `round(bytes / GiB, 2)`:
[mem_utils.py:30-31](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/utils/mem_utils.py#L30-L31).
Thus the printed number is neither exact integer bytes nor, for every path,
the final pool budget. A conservative rounding interval is the printed GiB
value plus/minus 0.005 GiB, converted outward to integer endpoints. Stripped
trailing zeros do not change that precision. Other-process memory changes and
ROCm fallback are explicitly handled at [W:713-732](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_worker.py#L713-L732).

## Stage 2: page geometry, groups and blocks

For a dense full-attention layer with block size `B`, per-rank KV heads `H`,
K/V dimensions `Dk,Dv` and storage width `s` bytes:

```text
layer_page_bytes = B * H * (Dk + Dv) * s
```

This reduces `num_heads * num_states * state_content_size_bytes` at
[I:538-560](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py#L538-L560),
with `num_states` at [I:228-234](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py#L228-L234).
Resolved `page_size_padded`, head slots, state content and tokens-per-state can
replace that simple geometry. Standard per-rank KV heads are
`max(1, total_KV_heads // TP)`, including replication when TP exceeds KV heads:
[config/model.py:1650-1670](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/model.py#L1650-L1670).

`auto` uses the runtime model dtype; ordinary FP8 spellings use one-byte storage:
[torch_utils.py:33-48](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/utils/torch_utils.py#L33-L48) and
[533-539](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/utils/torch_utils.py#L533-L539).
Weight quantization does not by itself select KV dtype. Per-tensor scales are
layer buffers, not an added per-page constant:
[attention.py:133-153](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/layers/attention/attention.py#L133-L153).
For Triton per-token-head FP8, content is K data plus V data plus two fp32
scales per head/state:
[triton_attn.py:284-295](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/attention/backends/triton_attn.py#L284-L295).
Other low-bit formats/backend packing must use their customized specs; dividing
all bytes by a nominal bit width is insufficient. MLA has no separate V plane
and can override packed content/alignment:
[I:676-702](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py#L676-L702).

Grouping is version-specific. Uniform specs use one group; uniform attention
types can aggregate differing pages. Hybrid, hidden-state, packed and fallback
paths are selected by [K:2159-2253](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L2159-L2253).
Shared layers do not receive independent cache specs:
[gpu_model_runner.py:7160-7179](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_model_runner.py#L7160-L7179).
A group's pages can alias physical pool storage with other groups; summing all
model layers into one unqualified cache bundle does not reproduce that planner.

For resolved device groups `g`, layer pages `p` and byte alignments `a`, let `S`
be bytes per shared-pool block. Layer-outer layouts use the largest group sum
of individually aligned pages. Block-outer layouts align the largest group sum
to the LCM of required alignments, including special HiSparse constraints:
[K:1515-1551](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L1515-L1551).
Then, for final available integer bytes `A`:

```text
N = A // S
N = configured_num_gpu_blocks_override, if supplied
```

Source: [K:1622-1629](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L1622-L1629).
The override is not the same contract as kvbench's budget-bounded override.
The common rank count is the minimum of per-worker counts, after global grouping
and PP projection: [K:2531-2553,2649-2678](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L2531-L2678).
Attention-free and HiSparse host/device cases have separate paths at
[K:1604-1620](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L1604-L1620).

## Stage 3: reported tokens and concurrency

For each device group, `r_g = ceil(group_max_request_bytes / group_page_bytes)`.
The source sums these request block widths across device groups. Host-resident
groups use their own pool and impose another minimum limit:
[K:1195-1227](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L1195-L1227).
For device-only groups and maximum model length `Lmax`:

```text
C = N / sum(r_g)                         # fractional float, not floor(C)
reported_tokens = int(C * Lmax)          # source float conversion/truncation
reported_concurrency = format(C, '.2f')
```

The token conversion and display are [K:2300-2334](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L2300-L2334).
For one ordinary full-attention group without DCP, `r_g=ceil(Lmax/B)`:
[I:602-607](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py#L602-L607).
Thus `N*B` and the displayed token capacity need not agree when `Lmax` is not
block-aligned. DCP changes full-attention request sizing in that same method.
For a sliding group, request width is
`ceil(min(window-1+extra_retained+max_in_flight,Lmax)/B)+1`:
[I:859-896](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py#L859-L896).
Uniform-type aggregate groups take the maximum layer request-page count:
[I:1253-1265](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py#L1253-L1265).

Startup admission reserves a null block, but allocation/reporting retains the
full count; auto-fit and override handling precede allocation:
[K:2605-2647](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L2605-L2647).
Reported maximum concurrency is an allocation ratio, not a runtime throughput
measurement or a promise that every other scheduler limit permits that count.

## Mapping to kvbench

References below use the candidate's unchanged capacity implementation.

| vLLM term/path | kvbench function and file:line | Agreement or gap and impact |
|---|---|---|
| Utilization budget | `estimate_budget`, [src/budget.cpp:15-25](../src/budget.cpp#L15-L25) | GAP: truncates; vLLM ceil can differ by a byte. |
| Resident weights | `estimate_weights`, [src/static_model.cpp:18-49](../src/static_model.cpp#L18-L49) | Explicit bytes or estimate; cannot reproduce profiling from config alone. |
| Non-Torch, activations, graphs | `estimate_budget`, [src/budget.cpp:26-35](../src/budget.cpp#L26-L35) | Inputs exist; GAP: additive planning terms do not derive current profiler's persistent-plus-transient result. |
| Manual available pool | Same budget subtraction | No dedicated claim-(a) input; an adapter can set total bytes to supplied pool, utilization one and every overhead zero. This bypasses claim (b), not validates it. |
| K/V pages and dtype bytes | `layer_data_bytes`, [src/memory.cpp:24-41](../src/memory.cpp#L24-L41) | Uniform dense/ordinary FP8 geometry maps when per-rank dimensions are supplied correctly. |
| Inline scale metadata | `layer_metadata_bytes`, [src/memory.cpp:44-55](../src/memory.cpp#L44-L55) | Configurable per-token-head bytes can map Triton scales; GAP: no automatic backend-specific packing selection. |
| Per-tensor scales | `cache_layout`, [src/memory.cpp:210-215](../src/memory.cpp#L210-L215) | GAP: configured constant pool metadata is not vLLM's already-profiled layer-buffer placement. Avoid double subtraction. |
| Layer sharing / TP replicas / PP projection | `cache_layout`, [src/memory.cpp:198-219](../src/memory.cpp#L198-L219); HF importer [src/integration.cpp:24-100](../src/integration.cpp#L24-L100) | GAP: no engine group/sharing/rank planner; direct HF import does not unwrap nested text configs or apply per-rank KV heads. |
| Hybrid groups / aligned physical pages | Same `cache_layout` | GAP: sums layer data; no largest-group aliased pool or backend alignment/group fallback inference. |
| Whole pool blocks | `pool_block_capacity`, [src/memory.cpp:240-244](../src/memory.cpp#L240-L244) | Maps single uniform group after metadata/geometry assumptions. |
| Block override | `estimate_budget`, [src/budget.cpp:38-42](../src/budget.cpp#L38-L42) | GAP: refuses overrides above budget; source vLLM accepts configured override. |
| Full request width | `sequence_pool_blocks`, [src/budget.cpp:8-9](../src/budget.cpp#L8-L9) | Maps one full group using context plus decode as Lmax. |
| Sliding request width | `blocks_unchecked`, [src/memory.cpp:59-63](../src/memory.cpp#L59-L63) | GAP: window cap plus configurable extra blocks omits version-specific in-flight/retained-token admission. |
| Reported token capacity | `estimate_budget`, [src/budget.cpp:43](../src/budget.cpp#L43) | GAP: `N*B`, versus vLLM request-rounded display above. |
| Reported concurrency | `estimate_budget`, [src/budget.cpp:44-46](../src/budget.cpp#L44-L46) | GAP: integer concurrent sequences, versus fractional reported ratio. |
| Log observations / residuals | `parse_vllm_log`, `compare_calibration`, [src/integration.cpp:103-235](../src/integration.cpp#L103-L235) | Memory/tokens/blocks comparison exists; GAP: does not parse/compare the fractional concurrency line or preserve memory-rounding intervals. |
| Frontend/MM reservations, host pools, null-block admission | No dedicated mapping | GAP: caller must supply effective final bytes/resolved specs; cannot infer all paths from the printed available-memory line. |

## Version compatibility and conditional log cross-checks

Older versions are not admitted by matching log wording alone. For the single
uniform full-attention path, v0.26.0 uses the same token conversion at
[568afb3:1819-1829](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/core/kv_cache_utils.py#L1819-L1829)
and an equivalent one-group concurrency divisor at lines 937-955.
v0.27.1 sums group request widths at
[6e448d0:937-959](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_utils.py#L937-L959).
v0.19.0 reports raw pool tokens at
[2a69949:1297-1327](https://github.com/vllm-project/vllm/blob/2a69949bdadf0e8942b7a1619b229cb475beef20/vllm/v1/core/kv_cache_utils.py#L1297-L1327);
this is only equivalent for the restricted full-attention, block-aligned,
non-context-parallel examples. Its hybrid output is not declared compatible.

The following are **conditional arithmetic cross-checks**, not complete
validation records. They assume block size 16, the pinned dense layout and the
captured model-config geometry; the logs do not establish every resolved input.
Two-decimal memory is converted to a point only to expose residuals. The range
uses the conservative memory-rounding interval, not a fitted overhead.

| Public log | Assumed geometry (layers/local KV heads/head dim; KV dtype; max length) | Printed GiB | Point tokens / reported | Rounding token range |
|---|---|---:|---:|---:|
| [#50705](https://github.com/vllm-project/vllm/issues/50705), Qwen2.5-0.5B | 24/2/64; FP8; 1024 | 55.32 | 9,667,856 / 9,668,416 | 9,666,992 to 9,668,736 |
| [#52802](https://github.com/vllm-project/vllm/issues/52802), Qwen3-0.6B | 28/8/128; bf16; 40960 | 13.86 | 129,760 / 129,744 | 129,712 to 129,808 |
| [#39010](https://github.com/vllm-project/vllm/issues/39010), Devstral Small | 40/4/128 (TP=2); FP8; 128000 | 11.87 | 311,152 / 311,264 | 311,024 to 311,296 |

All three reported counts lie inside those conditional ranges. This is not
exact reproduction, proof of the assumed block size, or evidence for the full
set of backend/group paths. The point residuals cannot justify tuning a model. Model-config inputs are
dated main-branch snapshots from the named Hugging Face models, not proof of
the exact historical checkpoint revisions.

## Exact and estimated claims

(a) **exact**: given vLLM's reported available KV cache memory, predict blocks,
tokens and max concurrency.

(b) **estimated**: predict available memory from config and hardware, with
stated uncertainty, since vLLM measures activation peak and non-torch memory
by profiling at startup.

For (a), an exact contract additionally requires integer effective pool bytes,
resolved layout/group/rank settings and the correct version's report semantics.
The rounded pre-reservation display alone does not establish that contract.
For (b), profiling-dependent inputs cannot be made exact by rearranging cache
arithmetic. Published-log validation is in progress; no general vLLM accuracy
claim or GPU timing calibration is established by this specification.
