#include "kvbench/static_model.hpp"

#include <algorithm>
#include <stdexcept>

namespace kvbench {
namespace {
Bytes packed_weights(Count parameters, CacheDType dtype) {
  const auto bits = dtype_bits(dtype);
  return bits == 4 ? ceil_div(parameters, 2) : checked_mul(parameters, bits / 8);
}
Count threshold(Bytes available, Count numerator, Count denominator) {
  return checked_add(checked_mul(available / denominator, numerator),
                     ceil_div(checked_mul(available % denominator, numerator), denominator));
}
}

WeightEstimate estimate_weights(const PlannerConfig& c) {
  validate_planner_config(c);
  if (c.weights.weights_bytes) return {checked_add(*c.weights.weights_bytes, c.weights.quantization_overhead_bytes), 0, false};
  const auto& a = c.cache.model;
  bool mla = a.attention_type == LayerKind::mla && a.layers.empty();
  for (const auto& layer : a.layers) mla = mla || layer.kind == LayerKind::mla;
  if (mla) throw std::invalid_argument("MLA weight shapes are not determined by cache dimensions; provide explicit weights_bytes or weights_gib");
  WeightEstimate result;
  result.estimated = true;
  const auto tensor = [&](Count parameters, Count copies = 1) {
    result.parameters = checked_add(result.parameters, checked_mul(parameters, copies));
    result.bytes = checked_add(result.bytes, checked_mul(packed_weights(parameters, c.weights.dtype), copies));
  };
  const Count query_width = checked_mul(a.num_attention_heads, a.head_dim);
  const Count kv_width = checked_mul(a.num_kv_heads, a.head_dim);
  tensor(checked_mul(a.hidden_size, c.weights.vocab_size), c.weights.tie_word_embeddings ? 1 : 2);
  tensor(a.hidden_size); // final RMS norm; biases excluded.
  tensor(checked_mul(a.hidden_size, query_width), checked_mul(a.num_layers, 2)); // Q and O.
  tensor(checked_mul(a.hidden_size, kv_width), checked_mul(a.num_layers, 2)); // K and V.
  const Count experts = checked_add(c.weights.num_experts, c.weights.num_shared_experts);
  tensor(checked_mul(a.hidden_size, c.weights.intermediate_size),
         checked_mul(checked_mul(a.num_layers, experts), c.weights.gated_mlp ? 3 : 2));
  tensor(a.hidden_size, checked_mul(a.num_layers, 2)); // pre-attention and pre-MLP RMS norms.
  if (c.weights.num_experts > 1) tensor(checked_mul(a.hidden_size, c.weights.num_experts), a.num_layers); // resident router.
  result.bytes = checked_add(result.bytes, c.weights.quantization_overhead_bytes);
  return result;
}

std::string memory_risk(Bytes required, Bytes available) {
  if (required > available) return "exceeded";
  if (required == available) return "at_capacity";
  if (required >= threshold(available, 19, 20)) return "critical";
  if (required >= threshold(available, 4, 5)) return "high";
  return "ok";
}

StaticEstimate estimate_static(const PlannerConfig& c, Bytes available_bytes,
                               std::optional<Count> max_concurrency) {
  validate_planner_config(c);
  if (max_concurrency && *max_concurrency == 0) throw std::invalid_argument("max_concurrency must be positive");
  const auto layout = cache_layout(c.cache);
  const Bytes prefill_per_sequence = sequence_cache_bytes(c.cache, c.workload.context_tokens);
  const Bytes decode_per_sequence = sequence_cache_bytes(c.cache, checked_add(c.workload.context_tokens, c.workload.decode_tokens));
  const Bytes fixed = checked_add(c.engine.activation_peak_bytes, layout.constant_metadata_bytes);
  StaticEstimate e;
  e.available_bytes = available_bytes;
  e.prefill_cache_bytes = checked_add(checked_mul(prefill_per_sequence, c.workload.concurrent_requests), layout.constant_metadata_bytes);
  e.decode_cache_bytes = checked_add(checked_mul(decode_per_sequence, c.workload.concurrent_requests), layout.constant_metadata_bytes);
  e.prefill_bytes = checked_add(e.prefill_cache_bytes, c.engine.activation_peak_bytes);
  e.decode_bytes = checked_add(e.decode_cache_bytes, c.engine.activation_peak_bytes);
  e.peak_bytes = std::max(e.prefill_bytes, e.decode_bytes);
  e.fits = e.peak_bytes <= available_bytes;
  e.risk = memory_risk(e.peak_bytes, available_bytes);
  if (available_bytes >= fixed) e.max_concurrency = checked_sub(available_bytes, fixed) / std::max(prefill_per_sequence, decode_per_sequence);
  if (max_concurrency && e.max_concurrency > *max_concurrency) {
    e.max_concurrency = *max_concurrency;
    e.concurrency_bounded = true;
  }
  return e;
}
}
