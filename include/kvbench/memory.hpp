#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kvbench {

using Bytes = std::uint64_t;
using Count = std::uint64_t;
inline constexpr Bytes gibibyte = Bytes{1} << 30;
inline constexpr Bytes gigabyte = 1'000'000'000;

/// Add byte/count operands or throw std::overflow_error; never wrap.
Count checked_add(Count a, Count b);
/// Multiply byte/count operands or throw std::overflow_error; never wrap.
Count checked_mul(Count a, Count b);
/// Subtract operands or throw std::underflow_error; never wrap.
Count checked_sub(Count a, Count b);
/// Ceiling division without overflowing a + divisor - 1; rejects zero divisor.
Count ceil_div(Count a, Count divisor);
/// Convert finite nonnegative GiB to whole bytes, rounding down; reject overflow.
Bytes gib_to_bytes(long double gib);
/// Convert bytes to GiB for presentation only; calculation stays integer.
long double bytes_to_gib(Bytes bytes);

enum class CacheDType { fp32, fp16, bf16, fp8_e4m3, fp8_e5m2, int8, int4 };
enum class LayerKind { full, sliding, mla };
enum class ScaleMode { none, per_tensor, per_token_head, group };

struct LayerSpec {
  LayerKind kind = LayerKind::full;
  Count window_tokens = 0;
};

struct Architecture {
  Count num_layers = 0;
  Count num_attention_heads = 0;
  Count num_kv_heads = 0;
  Count head_dim = 0;
  Count hidden_size = 0;
  Count kv_lora_rank = 0;
  Count qk_rope_head_dim = 0;
  LayerKind attention_type = LayerKind::full;
  Count sliding_window = 0;
  // Empty means the uniform attention_type repeated num_layers times.
  std::vector<LayerSpec> layers;
};

struct Quantization {
  CacheDType dtype = CacheDType::fp16;
  ScaleMode scale_mode = ScaleMode::none;
  Bytes scale_bytes = 4;
  Bytes zero_point_bytes = 0;
  Count group_size = 64;
};

struct CacheGeometry {
  Architecture model;
  Quantization quantization;
  Count block_size = 16;
  Count swa_extra_blocks = 1;
};

struct CacheLayout {
  Bytes data_bytes_per_token = 0;
  Bytes data_bytes_per_block = 0;
  Bytes metadata_bytes_per_block = 0;
  Bytes constant_metadata_bytes = 0;
  Bytes bytes_per_block = 0;
};

/// Return exact bits per element; reject an invalid enum value.
Count dtype_bits(CacheDType dtype);
/// Name of a supported cache dtype; reject invalid enum values.
std::string cache_dtype_name(CacheDType dtype);
/// Parse an exact cache dtype spelling, without lossy/implicit conversions.
CacheDType parse_cache_dtype(const std::string& name);
/// Validate dimensions, layer patterns, quantization and block bounds.
/// Non-divisible GQA heads produce a warning rather than a false support claim.
std::vector<std::string> validate_geometry(const CacheGeometry& geometry);
/// Derive integer bytes per block and constant scales, with checked arithmetic.
/// int4 packs each layer's K/V plane separately, padding its last half-byte.
CacheLayout cache_layout(const CacheGeometry& geometry);
/// Blocks for one layer: ceil(tokens/P), or ceil(min(tokens,window)/P)+extra.
/// Zero tokens allocate zero blocks; validates the entire geometry on entry.
Count layer_blocks(const CacheGeometry& geometry, Count layer_index, Count tokens);
/// Paged bytes for one sequence, summing independently capped layer allocations.
/// Excludes global per-tensor scales, which are allocated once per cache pool.
Bytes sequence_cache_bytes(const CacheGeometry& geometry, Count tokens);
/// Whole all-layer block bundles fitting the pool after constant metadata.
/// Hybrid layer budgets are aggregated; real engine group padding is not inferred.
Count pool_block_capacity(const CacheGeometry& geometry, Bytes pool_bytes);

}  // namespace kvbench
