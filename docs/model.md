# Cache memory model

The checked shared memory API is `include/kvbench/memory.hpp`, implemented in
`src/memory.cpp`. The CLI uses this shared API. All memory results are whole bytes (`uint64_t`), never floating GiB.
Presentation conversions use GiB = 2^30 bytes; decimal GB = 10^9 bytes.
Prototype formulas remain in Git history at ce2f280; they are not a supported contract.

## Arithmetic and storage

Addition, multiplication and subtraction throw on overflow/underflow. Ceiling
division is `a / b + (a % b != 0)`, avoiding the overflow in `a+b-1`.
GiB input converts to bytes by rounding down. Negative/non-finite values and
values at or above 2^64 bytes are rejected before integer conversion.

For standard attention, each layer stores two planes, K and V. Elements per
plane per token = `num_kv_heads * head_dim`; query heads do not multiply KV.
For MLA, there is one plane: `kv_lora_rank + qk_rope_head_dim` elements per
layer and token. This is latent caching; a backend expanding K/V uses more.
[The vLLM v0.10.1 implementation](https://github.com/vllm-project/vllm/blob/v0.10.1/vllm/v1/kv_cache_interface.py)
also distinguishes one MLA latent from two standard attention planes.

Storage bits per element: fp32=32, fp16/bf16=16,
fp8_e4m3/fp8_e5m2/int8=8, int4=4. Each layer/plane is packed separately.
At a block of P tokens, bytes per plane = `ceil(P * plane_elements * bits / 8)`.
For int4 this is computed as `ceil(elements/2)`, without multiplying into an
unrepresentable intermediate bit count. An odd final nibble pads to one byte.
`data_bytes_per_token` describes separately packed single-token planes, so it
need not multiply exactly into block bytes when a plane has odd int4 elements.

## Metadata

No metadata is included in `none` scale mode. For quantized storage, scale
bytes default to 4, zero-point bytes to 0 and group size to 64; select the actual
backend layout explicitly. These defaults are planning assumptions, not an
engine support or accuracy claim.

- `per_tensor`: one scale/zero-point entry per layer/plane, allocated once for
  the pool, rather than charged to every token or block.
- `per_token_head`: one entry per token/head/plane; MLA treats its latent as
  one head/plane. Metadata per block = P × heads × planes × entry bytes.
- `group`: one entry per group of N elements in each layer/plane/block,
  rounded up independently. Groups reset at each block boundary.

`bytes_per_block` sums all layer data and per-block metadata. Pool bundle
capacity = `floor((pool_bytes - constant_metadata_bytes)/bytes_per_block)`,
or zero when the pool cannot hold constant metadata. Actual quantization can
require alignment, zero points or additional metadata: configure and calibrate.

## Paging, sliding windows and hybrid layers

Full and MLA layers use `ceil(tokens/P)` blocks. Sliding layers use
`ceil(min(tokens,window)/P) + swa_extra_blocks` for nonzero tokens, defaulting
to one extra block. Zero tokens allocate no sequence blocks. Each layer can
have its own type/window; a supplied layer list must match `num_layers`.
Sequence bytes sum the individual layer block counts and byte widths.

The extra block is motivated by the window starting inside a block in
[the vLLM v0.10.1 SlidingWindowSpec](https://github.com/vllm-project/vllm/blob/v0.10.1/vllm/v1/kv_cache_interface.py).
That implementation also budgets newly scheduled prefill tokens. kvbench's
window cap plus configurable extra blocks is a planning abstraction, not an
exact reproduction of that version's chunk-admission formula. Real hybrid
cache-group padding and sharing are not inferred by this byte model.

## Golden derivations

These are mathematical expected values, verified by `tests/test_memory.cpp`,
not hardware measurements:

| Geometry, bf16 | Derivation | Bytes/token |
| --- | --- | ---: |
| Llama-3-8B-like GQA | 32 × 8 × 128 × 2 planes × 2 bytes | 131,072 |
| Same MHA | 32 × 32 × 128 × 2 × 2 | 524,288 |
| Qwen2.5-72B-like GQA | 80 × 8 × 128 × 2 × 2 | 327,680 |
| DeepSeek-V3-like latent MLA | 61 × (512+64) × 2 bytes | 70,272 |

At P=16, the first geometry requires 2,097,152 bytes/block. A 4 GiB pool
contains 2,048 blocks or 32,768 tokens, excluding weights/runtime overheads
(which belong to the budget phase). A 2,049-token sequence needs 129 blocks.
With int8/fp8 and no metadata, data is half bf16; int4 is one quarter when
packing dimensions are even. Tests separately verify exact scale overheads.

Every public memory entry point validates caller-mutated geometry. Dimensions
must be positive and bounded; invalid enums and layer patterns are errors.
Non-divisible KV/query head counts produce a warning for caller inspection.
