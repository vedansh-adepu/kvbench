#include "kvbench/budget.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kvbench {
Count sequence_pool_blocks(const CacheGeometry& geometry, Count tokens) {
  return ceil_div(sequence_cache_bytes(geometry, tokens), cache_layout(geometry).bytes_per_block);
}

BudgetResult estimate_budget(const PlannerConfig& c, std::optional<Count> max_concurrency) {
  validate_planner_config(c);
  BudgetResult b;
  b.total_gpu_bytes = c.total_gpu_bytes;
  // Avoid rounding UINT64_MAX to 2^64 on platforms with double-width long double.
  if (c.engine.gpu_memory_utilization == 1.0)
    b.utilization_bytes = c.total_gpu_bytes;
  else {
    const long double usable =
        static_cast<long double>(c.total_gpu_bytes) * c.engine.gpu_memory_utilization;
    if (!std::isfinite(usable) || usable >= std::ldexp(1.0L, 64))
      throw std::overflow_error("utilization budget exceeds uint64 bytes");
    b.utilization_bytes = static_cast<Bytes>(usable);
  }
  b.weights = estimate_weights(c);
  b.non_torch_overhead_bytes = c.engine.non_torch_overhead_bytes;
  b.activation_peak_bytes = c.engine.activation_peak_bytes;
  b.cudagraph_memory_bytes = c.engine.cudagraph_memory_bytes;
  const Bytes non_activation = checked_add(checked_add(b.weights.bytes, b.non_torch_overhead_bytes),
                                           b.cudagraph_memory_bytes);
  const Bytes all_overhead = checked_add(non_activation, b.activation_peak_bytes);
  b.overheads_fit = all_overhead <= b.utilization_bytes;
  b.overhead_deficit_bytes = b.overheads_fit ? 0 : checked_sub(all_overhead, b.utilization_bytes);
  b.kv_pool_bytes = b.overheads_fit ? checked_sub(b.utilization_bytes, all_overhead) : 0;
  b.cache = cache_layout(c.cache);
  b.num_blocks = pool_block_capacity(c.cache, b.kv_pool_bytes);
  if (c.engine.num_blocks) {
    if (*c.engine.num_blocks > b.num_blocks)
      throw std::invalid_argument("num_blocks override exceeds the memory budget");
    b.num_blocks = *c.engine.num_blocks;
  }
  b.max_tokens_in_cache = checked_mul(b.num_blocks, c.cache.block_size);
  b.blocks_per_sequence = sequence_pool_blocks(
      c.cache, checked_add(c.workload.context_tokens, c.workload.decode_tokens));
  b.max_concurrent_sequences = b.num_blocks / b.blocks_per_sequence;
  const Bytes pool_allocated_bytes = checked_mul(b.num_blocks, b.cache.bytes_per_block);
  const Bytes capacity_bytes =
      b.kv_pool_bytes < b.cache.constant_metadata_bytes
          ? b.kv_pool_bytes
          : checked_add(pool_allocated_bytes, b.cache.constant_metadata_bytes);
  const Bytes available =
      b.overheads_fit ? checked_add(capacity_bytes, b.activation_peak_bytes) : 0;
  b.workload = estimate_static(c, available, max_concurrency);
  // Aggregate pool bundles are the allocator unit. Uniform attention is exact;
  // heterogeneous layer bytes are rounded to that unit per sequence.
  const Count prefill_blocks = sequence_pool_blocks(c.cache, c.workload.context_tokens);
  b.workload.prefill_cache_bytes =
      checked_add(checked_mul(checked_mul(prefill_blocks, c.workload.concurrent_requests),
                              b.cache.bytes_per_block),
                  b.cache.constant_metadata_bytes);
  b.workload.decode_cache_bytes =
      checked_add(checked_mul(checked_mul(b.blocks_per_sequence, c.workload.concurrent_requests),
                              b.cache.bytes_per_block),
                  b.cache.constant_metadata_bytes);
  b.workload.prefill_bytes = checked_add(b.workload.prefill_cache_bytes, b.activation_peak_bytes);
  b.workload.decode_bytes = checked_add(b.workload.decode_cache_bytes, b.activation_peak_bytes);
  b.workload.peak_bytes = std::max(b.workload.prefill_bytes, b.workload.decode_bytes);
  b.workload.available_bytes = available;
  b.workload.fits = b.workload.peak_bytes <= available;
  b.workload.risk = memory_risk(b.workload.peak_bytes, available);
  if (max_concurrency && b.max_concurrent_sequences > *max_concurrency) {
    b.max_concurrent_sequences = *max_concurrency;
    b.workload.concurrency_bounded = true;
  } else
    b.workload.concurrency_bounded = false;
  b.workload.max_concurrency = b.max_concurrent_sequences;
  return b;
}
}  // namespace kvbench
