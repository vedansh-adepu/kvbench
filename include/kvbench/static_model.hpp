#pragma once
#include <optional>
#include <string>
#include "kvbench/planner_config.hpp"

namespace kvbench {
struct WeightEstimate {
  Bytes bytes = 0;
  Count parameters = 0;
  bool estimated = false;
};
struct StaticEstimate {
  Bytes prefill_cache_bytes = 0;
  Bytes decode_cache_bytes = 0;
  Bytes prefill_bytes = 0;
  Bytes decode_bytes = 0;
  Bytes peak_bytes = 0;
  Bytes available_bytes = 0;
  Count max_concurrency = 0;
  bool concurrency_bounded = false;
  bool fits = false;
  std::string risk;
};
/// Estimate dense/GQA/MoE resident weights in bytes with per-tensor int4 padding.
/// Explicit weights bypass estimation; MLA requires explicit checkpoint bytes
/// because latent cache dimensions do not determine its projection shapes.
WeightEstimate estimate_weights(const PlannerConfig& config);
/// Classify required/available bytes with consistent <= fit semantics.
std::string memory_risk(Bytes required, Bytes available);
/// Static phase-correct paged cache + fixed activation bytes, excluding weights.
/// available_bytes is the budget for cache and activation together. A supplied
/// max_concurrency bound labels the returned capacity as a lower bound only
/// when the true capacity is greater. Never silently cap or wrap capacity.
StaticEstimate estimate_static(const PlannerConfig& config, Bytes available_bytes,
                               std::optional<Count> max_concurrency = std::nullopt);
}
