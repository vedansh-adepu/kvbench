#pragma once

#include <optional>
#include <string>
#include <vector>

#include "kvbench/memory.hpp"

namespace kvbench {
struct WeightSpec {
  std::optional<Bytes> weights_bytes;
  CacheDType dtype = CacheDType::bf16;
  Count intermediate_size = 0;
  Count vocab_size = 32000;
  bool tie_word_embeddings = true;
  bool gated_mlp = true;
  Count num_experts = 1;
  Count num_shared_experts = 0;
  Bytes quantization_overhead_bytes = 0;
};
struct EngineSpec {
  double gpu_memory_utilization = 0.9;
  Bytes non_torch_overhead_bytes = gibibyte / 4;
  Bytes activation_peak_bytes = gibibyte;
  Bytes cudagraph_memory_bytes = gibibyte / 2;
  Count block_size = 16;
  Count swa_extra_blocks = 1;
  Count max_num_seqs = 256;
  Count max_num_batched_tokens = 2048;
  bool enable_chunked_prefill = true;
  std::optional<Count> num_blocks;
  std::string preemption_mode = "recompute";
  double base_step_ms = 1;
  double prefill_ms_per_token = 0.01;
  double decode_ms_per_seq = 0.1;
  double max_simulated_time_ms = 86'400'000;
  Count max_steps = 1'000'000;
  bool calibrated = false;
};
struct ArrivalRequest {
  std::string id;
  double arrival_ms = 0;
  Count input_tokens = 0;
  Count output_tokens = 0;
};
struct StaticWorkload {
  Count context_tokens = 1;
  Count decode_tokens = 0;
  Count concurrent_requests = 1;
  std::vector<ArrivalRequest> requests;
};
struct PlannerConfig {
  int schema_version = 2;
  std::string model_name = "unnamed-model";
  CacheGeometry cache;
  WeightSpec weights;
  EngineSpec engine;
  Bytes total_gpu_bytes = Bytes{24} * gibibyte;
  StaticWorkload workload;
  std::vector<std::string> warnings;
};
/// Parse strict schema v2, or map a schema-less v1 document with warnings.
/// No IO or stderr output; warnings are returned for the CLI to print.
PlannerConfig parse_planner_config(const std::string& source);
/// Load at most 16 MiB and parse strict configuration; no writes.
PlannerConfig load_planner_config(const std::string& path);
/// Validate mutable public configs, ranges and all checked count sums.
void validate_planner_config(const PlannerConfig& config);
}  // namespace kvbench
