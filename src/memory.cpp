#include "kvbench/memory.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kvbench {
namespace {

void positive(Count value, Count maximum, const char* field) {
  if (value == 0 || value > maximum) {
    throw std::invalid_argument(std::string(field) +
                                " must be positive and <= " + std::to_string(maximum));
  }
}

LayerSpec layer_at(const Architecture& model, Count index) {
  if (index >= model.num_layers) throw std::out_of_range("layer index outside architecture");
  if (model.layers.empty()) return {model.attention_type, model.sliding_window};
  return model.layers.at(static_cast<std::size_t>(index));
}

Count planes(const LayerSpec& layer) { return layer.kind == LayerKind::mla ? 1 : 2; }

Count plane_elements(const Architecture& model, const LayerSpec& layer) {
  if (layer.kind == LayerKind::mla) return checked_add(model.kv_lora_rank, model.qk_rope_head_dim);
  return checked_mul(model.num_kv_heads, model.head_dim);
}

Bytes packed_plane_bytes(Count elements, CacheDType dtype) {
  // Divide before multiplying to avoid overflowing the intermediate bit count
  // when the resulting byte count is representable.
  const Count bits = dtype_bits(dtype);
  if (bits == 4) return ceil_div(elements, 2);
  return checked_mul(elements, bits / 8);
}

Bytes layer_data_bytes(const CacheGeometry& geometry, const LayerSpec& layer, Count tokens) {
  const Count elements = checked_mul(plane_elements(geometry.model, layer), tokens);
  return checked_mul(planes(layer), packed_plane_bytes(elements, geometry.quantization.dtype));
}

Bytes layer_metadata_bytes(const CacheGeometry& geometry, const LayerSpec& layer, Count tokens) {
  const auto& q = geometry.quantization;
  if (q.scale_mode == ScaleMode::none || q.scale_mode == ScaleMode::per_tensor) return 0;
  const Bytes entry = checked_add(q.scale_bytes, q.zero_point_bytes);
  if (q.scale_mode == ScaleMode::per_token_head) {
    const Count heads = layer.kind == LayerKind::mla ? 1 : geometry.model.num_kv_heads;
    return checked_mul(checked_mul(checked_mul(tokens, heads), planes(layer)), entry);
  }
  if (q.scale_mode == ScaleMode::group) {
    const Count elements = checked_mul(plane_elements(geometry.model, layer), tokens);
    return checked_mul(checked_mul(ceil_div(elements, q.group_size), planes(layer)), entry);
  }
  return 0;
}

Count blocks_unchecked(const CacheGeometry& geometry, const LayerSpec& layer, Count tokens) {
  if (tokens == 0) return 0;
  if (layer.kind != LayerKind::sliding) return ceil_div(tokens, geometry.block_size);
  return checked_add(ceil_div(std::min(tokens, layer.window_tokens), geometry.block_size),
                     geometry.swa_extra_blocks);
}

}  // namespace

Count checked_add(Count a, Count b) {
  if (b > std::numeric_limits<Count>::max() - a)
    throw std::overflow_error("integer addition overflow");
  return a + b;
}

Count checked_mul(Count a, Count b) {
  if (a != 0 && b > std::numeric_limits<Count>::max() / a)
    throw std::overflow_error("integer multiplication overflow");
  return a * b;
}

Count checked_sub(Count a, Count b) {
  if (b > a) throw std::underflow_error("integer subtraction underflow");
  return a - b;
}

Count ceil_div(Count a, Count divisor) {
  if (divisor == 0) throw std::invalid_argument("division by zero");
  return a / divisor + static_cast<Count>(a % divisor != 0);
}

Bytes gib_to_bytes(long double gib) {
  const long double value = gib * static_cast<long double>(gibibyte);
  // 2^64 is an exact exclusive bound even where long double is just double.
  if (!std::isfinite(gib) || gib < 0 || !std::isfinite(value) || value >= std::ldexp(1.0L, 64)) {
    throw std::invalid_argument("GiB value must be finite, nonnegative and fit uint64 bytes");
  }
  return static_cast<Bytes>(value);
}

long double bytes_to_gib(Bytes bytes) {
  return static_cast<long double>(bytes) / static_cast<long double>(gibibyte);
}

Count dtype_bits(CacheDType dtype) {
  switch (dtype) {
    case CacheDType::fp32:
      return 32;
    case CacheDType::fp16:
    case CacheDType::bf16:
      return 16;
    case CacheDType::fp8_e4m3:
    case CacheDType::fp8_e5m2:
    case CacheDType::int8:
      return 8;
    case CacheDType::int4:
      return 4;
  }
  throw std::invalid_argument("invalid cache dtype enum");
}

std::string cache_dtype_name(CacheDType dtype) {
  switch (dtype) {
    case CacheDType::fp32:
      return "fp32";
    case CacheDType::fp16:
      return "fp16";
    case CacheDType::bf16:
      return "bf16";
    case CacheDType::fp8_e4m3:
      return "fp8_e4m3";
    case CacheDType::fp8_e5m2:
      return "fp8_e5m2";
    case CacheDType::int8:
      return "int8";
    case CacheDType::int4:
      return "int4";
  }
  throw std::invalid_argument("invalid cache dtype enum");
}

CacheDType parse_cache_dtype(const std::string& name) {
  for (const auto value :
       {CacheDType::fp32, CacheDType::fp16, CacheDType::bf16, CacheDType::fp8_e4m3,
        CacheDType::fp8_e5m2, CacheDType::int8, CacheDType::int4}) {
    if (name == cache_dtype_name(value)) return value;
  }
  throw std::invalid_argument("unsupported cache dtype: " + name);
}

std::vector<std::string> validate_geometry(const CacheGeometry& geometry) {
  const auto& model = geometry.model;
  positive(model.num_layers, 4096, "num_layers");
  positive(model.num_attention_heads, 65536, "num_attention_heads");
  positive(model.num_kv_heads, 65536, "num_kv_heads");
  positive(model.head_dim, 1048576, "head_dim");
  positive(model.hidden_size, 16777216, "hidden_size");
  positive(geometry.block_size, 1048576, "block_size");
  if (geometry.swa_extra_blocks > 1048576)
    throw std::invalid_argument("swa_extra_blocks exceeds maximum");
  if (!model.layers.empty() && model.layers.size() != model.num_layers)
    throw std::invalid_argument("layer pattern length must equal num_layers");
  static_cast<void>(dtype_bits(geometry.quantization.dtype));
  const auto& q = geometry.quantization;
  if (q.scale_mode != ScaleMode::none && q.scale_mode != ScaleMode::per_tensor &&
      q.scale_mode != ScaleMode::per_token_head && q.scale_mode != ScaleMode::group)
    throw std::invalid_argument("invalid scale mode");
  if (q.scale_mode != ScaleMode::none) {
    positive(q.scale_bytes, 16, "scale_bytes");
    if (q.zero_point_bytes > 16) throw std::invalid_argument("zero_point_bytes exceeds maximum");
    if (dtype_bits(q.dtype) > 8)
      throw std::invalid_argument("scale metadata requires a quantized cache dtype");
  }
  if (q.scale_mode == ScaleMode::group) positive(q.group_size, 16777216, "group_size");
  for (Count i = 0; i < model.num_layers; ++i) {
    const auto layer = layer_at(model, i);
    switch (layer.kind) {
      case LayerKind::full:
        break;
      case LayerKind::sliding:
        if (layer.window_tokens == 0)
          throw std::invalid_argument("sliding layer requires a positive window");
        break;
      case LayerKind::mla:
        positive(model.kv_lora_rank, 1048576, "kv_lora_rank");
        positive(model.qk_rope_head_dim, 1048576, "qk_rope_head_dim");
        break;
      default:
        throw std::invalid_argument("invalid layer kind");
    }
  }
  std::vector<std::string> warnings;
  if (model.num_kv_heads > model.num_attention_heads ||
      model.num_attention_heads % model.num_kv_heads != 0)
    warnings.emplace_back(
        "num_kv_heads does not divide num_attention_heads; verify architecture and engine support");
  return warnings;
}

CacheLayout cache_layout(const CacheGeometry& geometry) {
  static_cast<void>(validate_geometry(geometry));
  CacheLayout layout;
  for (Count i = 0; i < geometry.model.num_layers; ++i) {
    const auto layer = layer_at(geometry.model, i);
    layout.data_bytes_per_token =
        checked_add(layout.data_bytes_per_token, layer_data_bytes(geometry, layer, 1));
    layout.data_bytes_per_block = checked_add(
        layout.data_bytes_per_block, layer_data_bytes(geometry, layer, geometry.block_size));
    layout.metadata_bytes_per_block =
        checked_add(layout.metadata_bytes_per_block,
                    layer_metadata_bytes(geometry, layer, geometry.block_size));
    if (geometry.quantization.scale_mode == ScaleMode::per_tensor) {
      const Bytes entry =
          checked_add(geometry.quantization.scale_bytes, geometry.quantization.zero_point_bytes);
      layout.constant_metadata_bytes =
          checked_add(layout.constant_metadata_bytes, checked_mul(planes(layer), entry));
    }
  }
  layout.bytes_per_block =
      checked_add(layout.data_bytes_per_block, layout.metadata_bytes_per_block);
  return layout;
}

Count layer_blocks(const CacheGeometry& geometry, Count layer_index, Count tokens) {
  static_cast<void>(validate_geometry(geometry));
  return blocks_unchecked(geometry, layer_at(geometry.model, layer_index), tokens);
}

Bytes sequence_cache_bytes(const CacheGeometry& geometry, Count tokens) {
  static_cast<void>(validate_geometry(geometry));
  Bytes bytes = 0;
  for (Count i = 0; i < geometry.model.num_layers; ++i) {
    const auto layer = layer_at(geometry.model, i);
    const Bytes block_bytes =
        checked_add(layer_data_bytes(geometry, layer, geometry.block_size),
                    layer_metadata_bytes(geometry, layer, geometry.block_size));
    bytes = checked_add(bytes, checked_mul(blocks_unchecked(geometry, layer, tokens), block_bytes));
  }
  return bytes;
}

Count pool_block_capacity(const CacheGeometry& geometry, Bytes pool_bytes) {
  const auto layout = cache_layout(geometry);
  if (pool_bytes < layout.constant_metadata_bytes) return 0;
  return checked_sub(pool_bytes, layout.constant_metadata_bytes) / layout.bytes_per_block;
}

}  // namespace kvbench
