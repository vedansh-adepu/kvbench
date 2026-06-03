#pragma once

#include <string>
#include <vector>

#include "kvbench/config.hpp"

namespace kvbench {

struct Estimate {
  std::string model_name;
  int layers = 0;
  int attention_heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int hidden_size = 0;
  std::string model_dtype;
  std::string kv_dtype;
  int context_tokens = 0;
  int decode_tokens = 0;
  int concurrent_requests = 0;
  int batch_size = 0;
  int prefill_chunk_size = 0;
  double per_token_kv_bytes = 0.0;
  double kv_cache_bytes = 0.0;
  double kv_fragmentation_bytes = 0.0;
  double prefill_bytes = 0.0;
  double decode_bytes = 0.0;
  double peak_working_bytes = 0.0;
  double gpu_memory_bytes = 0.0;
  double reserved_memory_bytes = 0.0;
  double usable_memory_bytes = 0.0;
  double total_with_reserved_bytes = 0.0;
  double headroom_bytes = 0.0;
  int recommended_max_concurrency = 0;
  double throughput_pressure_score = 0.0;
  std::string oom_risk;
  bool fits_usable_memory = false;
};

struct SweepRow {
  int batch = 0;
  double peak_gb = 0.0;
  double headroom_gb = 0.0;
  std::string risk;
  bool fits = false;
};

double bytes_to_gb(double bytes);
double gb_to_bytes(double gb);

Estimate estimate(const Config& config);
std::vector<SweepRow> sweep_batches(const Config& base, int batch_min, int batch_max);

std::string format_text_report(const Estimate& estimate);
std::string format_json_report(const Estimate& estimate);
std::string format_markdown_report(const Estimate& estimate);
std::string format_compare_table(const std::vector<std::pair<std::string, Estimate>>& estimates);
std::string format_sweep_table(const std::vector<SweepRow>& rows);
std::string format_sweep_csv(const std::vector<SweepRow>& rows);
std::string format_mermaid_graph(const Estimate& estimate);

}  // namespace kvbench
