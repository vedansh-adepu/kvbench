#pragma once
#include "kvbench/static_model.hpp"

namespace kvbench {
struct BudgetResult {
  Bytes total_gpu_bytes = 0;
  Bytes utilization_bytes = 0;
  WeightEstimate weights;
  Bytes non_torch_overhead_bytes = 0;
  Bytes activation_peak_bytes = 0;
  Bytes cudagraph_memory_bytes = 0;
  Bytes overhead_deficit_bytes = 0;
  Bytes kv_pool_bytes = 0;
  CacheLayout cache;
  Count num_blocks = 0;
  Count max_tokens_in_cache = 0;
  Count blocks_per_sequence = 0;
  Count max_concurrent_sequences = 0;
  bool overheads_fit = false;
  StaticEstimate workload;
};
/// vLLM-style memory budget in checked uint64 bytes. Utilization rounds down;
/// subtract weights and each explicit overhead once. Insufficient budget yields
/// zero KV pool and a deficit, not unsigned wrap. Block override must fit pool.
BudgetResult estimate_budget(const PlannerConfig& config,
                             std::optional<Count> max_concurrency = std::nullopt);
/// Return whole aggregate pool bundles required for one paged sequence.
/// Hybrid allocations round up the sum of independently capped layer bytes.
Count sequence_pool_blocks(const CacheGeometry& geometry, Count tokens);
}
