#include "kvbench/estimator.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "kvbench/json.hpp"
#include "kvbench/scheduler.hpp"

namespace kvbench {
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

long long round_up_to_page(long long tokens, int page_size) {
  if (page_size <= 0) return tokens;
  return ((tokens + page_size - 1) / page_size) * page_size;
}

std::string fixed(double value, int precision = 2) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

std::string risk_for(double ratio) {
  if (ratio >= 1.0) return "exceeded";
  if (ratio >= 0.90) return "high";
  if (ratio >= 0.75) return "elevated";
  if (ratio >= 0.60) return "moderate";
  return "low";
}

double estimate_prefill_scratch_bytes(const Config& config) {
  const int chunk = std::min(config.workload.context_tokens, config.workload.prefill_chunk_size);
  const int active_batch = std::min(config.workload.batch_size, config.workload.concurrent_requests);
  return static_cast<double>(config.model.layers) *
         static_cast<double>(config.model.hidden_size) *
         static_cast<double>(std::max(1, chunk)) *
         static_cast<double>(std::max(1, active_batch)) *
         dtype_bytes(config.model.dtype) * 0.25;
}

double estimate_decode_scratch_bytes(const Config& config) {
  return static_cast<double>(config.model.layers) *
         static_cast<double>(config.model.hidden_size) *
         static_cast<double>(std::max(1, config.workload.batch_size)) *
         dtype_bytes(config.model.dtype) * 0.25;
}

Config with_concurrency(Config config, int concurrency) {
  config.workload.concurrent_requests = concurrency;
  config.workload.batch_size = concurrency;
  config.requests.clear();
  return config;
}

double peak_bytes_for_concurrency(const Config& config, double per_token_kv_bytes, int concurrency) {
  const long long tokens_per_request = static_cast<long long>(config.workload.context_tokens) +
                                       static_cast<long long>(config.workload.decode_tokens);
  const long long paged_tokens_per_request = round_up_to_page(tokens_per_request, config.system.page_size_tokens);
  const double fragmentation_per_request =
      per_token_kv_bytes * static_cast<double>(paged_tokens_per_request - tokens_per_request);
  const int chunk = std::min(config.workload.context_tokens, config.workload.prefill_chunk_size);
  const double prefill_scratch =
      static_cast<double>(config.model.layers) *
      static_cast<double>(config.model.hidden_size) *
      static_cast<double>(std::max(1, chunk)) *
      static_cast<double>(std::max(1, concurrency)) *
      dtype_bytes(config.model.dtype) * 0.25;
  const double decode_scratch =
      static_cast<double>(config.model.layers) *
      static_cast<double>(config.model.hidden_size) *
      static_cast<double>(std::max(1, concurrency)) *
      dtype_bytes(config.model.dtype) * 0.25;
  const double prefill = per_token_kv_bytes *
                             static_cast<double>(config.workload.context_tokens) *
                             static_cast<double>(concurrency) +
                         fragmentation_per_request * static_cast<double>(concurrency) +
                         prefill_scratch;
  const double decode = per_token_kv_bytes *
                            static_cast<double>(tokens_per_request) *
                            static_cast<double>(concurrency) +
                        fragmentation_per_request * static_cast<double>(concurrency) +
                        decode_scratch;
  return std::max(prefill, decode);
}

}  // namespace

double bytes_to_gb(double bytes) {
  return bytes / kGiB;
}

double gb_to_bytes(double gb) {
  return gb * kGiB;
}

Estimate estimate(const Config& config) {
  validate_config(config, "estimate");
  Estimate result;
  result.model_name = config.model.name;
  result.layers = config.model.layers;
  result.attention_heads = config.model.attention_heads;
  result.kv_heads = config.model.kv_heads;
  result.head_dim = config.model.head_dim;
  result.hidden_size = config.model.hidden_size;
  result.model_dtype = dtype_name(config.model.dtype);
  result.kv_dtype = dtype_name(config.system.kv_quantization);
  result.context_tokens = config.workload.context_tokens;
  result.decode_tokens = config.workload.decode_tokens;
  result.concurrent_requests = config.workload.concurrent_requests;
  result.batch_size = config.workload.batch_size;
  result.prefill_chunk_size = config.workload.prefill_chunk_size;

  result.per_token_kv_bytes = static_cast<double>(config.model.layers) *
                              static_cast<double>(config.model.kv_heads) *
                              static_cast<double>(config.model.head_dim) *
                              2.0 *
                              dtype_bytes(config.system.kv_quantization);

  const long long tokens_per_request = static_cast<long long>(config.workload.context_tokens) +
                                       static_cast<long long>(config.workload.decode_tokens);
  const long long paged_tokens_per_request = round_up_to_page(tokens_per_request, config.system.page_size_tokens);
  const long long active_tokens = tokens_per_request * static_cast<long long>(config.workload.concurrent_requests);
  const long long paged_active_tokens = paged_tokens_per_request * static_cast<long long>(config.workload.concurrent_requests);

  result.kv_cache_bytes = result.per_token_kv_bytes * static_cast<double>(active_tokens);
  result.kv_fragmentation_bytes = result.per_token_kv_bytes * static_cast<double>(paged_active_tokens - active_tokens);
  result.prefill_bytes = result.per_token_kv_bytes *
                             static_cast<double>(config.workload.context_tokens) *
                             static_cast<double>(config.workload.concurrent_requests) +
                         result.kv_fragmentation_bytes +
                         estimate_prefill_scratch_bytes(config);
  result.decode_bytes = result.kv_cache_bytes +
                        result.kv_fragmentation_bytes +
                        estimate_decode_scratch_bytes(config);
  result.peak_working_bytes = std::max(result.prefill_bytes, result.decode_bytes);

  if (!config.requests.empty()) {
    const SchedulerResult scheduler = simulate_scheduler(config);
    result.peak_working_bytes = std::max(result.peak_working_bytes, scheduler.peak_memory_bytes);
  }

  result.gpu_memory_bytes = gb_to_bytes(config.system.gpu_memory_gb);
  result.reserved_memory_bytes = gb_to_bytes(config.system.reserved_memory_gb);
  result.usable_memory_bytes = result.gpu_memory_bytes - result.reserved_memory_bytes;
  result.total_with_reserved_bytes = result.peak_working_bytes + result.reserved_memory_bytes;
  result.headroom_bytes = result.usable_memory_bytes - result.peak_working_bytes;
  result.fits_usable_memory = result.headroom_bytes >= 0.0;
  result.oom_risk = risk_for(result.peak_working_bytes / result.usable_memory_bytes);

  int low = 0;
  int high = 1;
  while (high < 100000 &&
         peak_bytes_for_concurrency(config, result.per_token_kv_bytes, high) <= result.usable_memory_bytes) {
    low = high;
    high *= 2;
  }
  high = std::min(high, 100000);
  while (low + 1 < high) {
    const int mid = low + (high - low) / 2;
    if (peak_bytes_for_concurrency(config, result.per_token_kv_bytes, mid) <= result.usable_memory_bytes) {
      low = mid;
    } else {
      high = mid;
    }
  }
  result.recommended_max_concurrency = low;

  const double token_work = static_cast<double>(config.workload.context_tokens) +
                            static_cast<double>(config.workload.decode_tokens) * 4.0;
  result.throughput_pressure_score = std::min(100.0,
      (static_cast<double>(config.workload.batch_size) * token_work / 8192.0) +
      (result.peak_working_bytes / result.usable_memory_bytes) * 50.0);
  return result;
}

std::vector<SweepRow> sweep_batches(const Config& base, int batch_min, int batch_max) {
  std::vector<SweepRow> rows;
  for (int batch = batch_min; batch <= batch_max; ++batch) {
    Config config = with_concurrency(base, batch);
    const Estimate current = estimate(config);
    rows.push_back({batch, bytes_to_gb(current.peak_working_bytes), bytes_to_gb(current.headroom_bytes), current.oom_risk, current.fits_usable_memory});
  }
  return rows;
}

std::string format_text_report(const Estimate& e) {
  std::ostringstream out;
  out << "kvbench simulation report\n";
  out << "model: " << e.model_name << "\n";
  out << "layers: " << e.layers << "\n";
  out << "attention heads: " << e.attention_heads << "\n";
  out << "kv heads: " << e.kv_heads << "\n";
  out << "head dimension: " << e.head_dim << "\n";
  out << "model dtype: " << e.model_dtype << "\n";
  out << "kv dtype: " << e.kv_dtype << "\n";
  out << "context tokens: " << e.context_tokens << "\n";
  out << "decode tokens: " << e.decode_tokens << "\n";
  out << "concurrent requests: " << e.concurrent_requests << "\n";
  out << "batch size: " << e.batch_size << "\n";
  out << "per-token KV bytes: " << fixed(e.per_token_kv_bytes, 0) << "\n";
  out << "KV-cache memory estimate: " << fixed(bytes_to_gb(e.kv_cache_bytes + e.kv_fragmentation_bytes)) << " GiB\n";
  out << "prefill memory estimate: " << fixed(bytes_to_gb(e.prefill_bytes)) << " GiB\n";
  out << "decode memory estimate: " << fixed(bytes_to_gb(e.decode_bytes)) << " GiB\n";
  out << "usable memory budget: " << fixed(bytes_to_gb(e.usable_memory_bytes)) << " GiB\n";
  out << "reserved memory: " << fixed(bytes_to_gb(e.reserved_memory_bytes)) << " GiB\n";
  out << "memory headroom: " << fixed(bytes_to_gb(e.headroom_bytes)) << " GiB\n";
  out << "OOM risk: " << e.oom_risk << "\n";
  out << "recommended max concurrency: " << e.recommended_max_concurrency << "\n";
  out << "throughput pressure score: " << fixed(e.throughput_pressure_score, 1) << " / 100 heuristic\n";
  return out.str();
}

std::string format_json_report(const Estimate& e) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"model_name\": \"" << escape_json_string(e.model_name) << "\",\n";
  out << "  \"layers\": " << e.layers << ",\n";
  out << "  \"attention_heads\": " << e.attention_heads << ",\n";
  out << "  \"kv_heads\": " << e.kv_heads << ",\n";
  out << "  \"head_dim\": " << e.head_dim << ",\n";
  out << "  \"model_dtype\": \"" << e.model_dtype << "\",\n";
  out << "  \"kv_dtype\": \"" << e.kv_dtype << "\",\n";
  out << "  \"context_tokens\": " << e.context_tokens << ",\n";
  out << "  \"decode_tokens\": " << e.decode_tokens << ",\n";
  out << "  \"concurrent_requests\": " << e.concurrent_requests << ",\n";
  out << "  \"batch_size\": " << e.batch_size << ",\n";
  out << "  \"kv_cache_gb\": " << fixed(bytes_to_gb(e.kv_cache_bytes + e.kv_fragmentation_bytes), 6) << ",\n";
  out << "  \"prefill_gb\": " << fixed(bytes_to_gb(e.prefill_bytes), 6) << ",\n";
  out << "  \"decode_gb\": " << fixed(bytes_to_gb(e.decode_bytes), 6) << ",\n";
  out << "  \"usable_memory_gb\": " << fixed(bytes_to_gb(e.usable_memory_bytes), 6) << ",\n";
  out << "  \"headroom_gb\": " << fixed(bytes_to_gb(e.headroom_bytes), 6) << ",\n";
  out << "  \"oom_risk\": \"" << e.oom_risk << "\",\n";
  out << "  \"fits_usable_memory\": " << (e.fits_usable_memory ? "true" : "false") << ",\n";
  out << "  \"recommended_max_concurrency\": " << e.recommended_max_concurrency << ",\n";
  out << "  \"throughput_pressure_score\": " << fixed(e.throughput_pressure_score, 3) << "\n";
  out << "}\n";
  return out.str();
}

std::string format_markdown_report(const Estimate& e) {
  std::ostringstream out;
  out << "# kvbench Simulation Report\n\n";
  out << "| Metric | Value |\n|---|---:|\n";
  out << "| Model | " << e.model_name << " |\n";
  out << "| Layers | " << e.layers << " |\n";
  out << "| Attention heads | " << e.attention_heads << " |\n";
  out << "| KV heads | " << e.kv_heads << " |\n";
  out << "| Head dimension | " << e.head_dim << " |\n";
  out << "| Model dtype | " << e.model_dtype << " |\n";
  out << "| KV dtype | " << e.kv_dtype << " |\n";
  out << "| Context tokens | " << e.context_tokens << " |\n";
  out << "| Decode tokens | " << e.decode_tokens << " |\n";
  out << "| Concurrent requests | " << e.concurrent_requests << " |\n";
  out << "| KV-cache memory | " << fixed(bytes_to_gb(e.kv_cache_bytes + e.kv_fragmentation_bytes)) << " GiB |\n";
  out << "| Prefill memory | " << fixed(bytes_to_gb(e.prefill_bytes)) << " GiB |\n";
  out << "| Decode memory | " << fixed(bytes_to_gb(e.decode_bytes)) << " GiB |\n";
  out << "| OOM risk | " << e.oom_risk << " |\n";
  out << "| Recommended max concurrency | " << e.recommended_max_concurrency << " |\n";
  return out.str();
}

std::string format_compare_table(const std::vector<std::pair<std::string, Estimate>>& estimates) {
  std::ostringstream out;
  out << "config\tmodel\tconcurrency\tkv_gb\tpeak_gb\theadroom_gb\trisk\tmax_concurrency\n";
  for (const auto& [name, e] : estimates) {
    out << name << '\t'
        << e.model_name << '\t'
        << e.concurrent_requests << '\t'
        << fixed(bytes_to_gb(e.kv_cache_bytes + e.kv_fragmentation_bytes)) << '\t'
        << fixed(bytes_to_gb(e.peak_working_bytes)) << '\t'
        << fixed(bytes_to_gb(e.headroom_bytes)) << '\t'
        << e.oom_risk << '\t'
        << e.recommended_max_concurrency << '\n';
  }
  return out.str();
}

std::string format_sweep_table(const std::vector<SweepRow>& rows) {
  std::ostringstream out;
  out << "batch\tpeak_gb\theadroom_gb\trisk\tfits\n";
  for (const auto& row : rows) {
    out << row.batch << '\t' << fixed(row.peak_gb) << '\t' << fixed(row.headroom_gb) << '\t'
        << row.risk << '\t' << (row.fits ? "yes" : "no") << '\n';
  }
  return out.str();
}

std::string format_sweep_csv(const std::vector<SweepRow>& rows) {
  std::ostringstream out;
  out << "batch,peak_gb,headroom_gb,risk,fits\n";
  for (const auto& row : rows) {
    out << row.batch << ',' << fixed(row.peak_gb) << ',' << fixed(row.headroom_gb) << ','
        << row.risk << ',' << (row.fits ? "true" : "false") << '\n';
  }
  return out.str();
}

std::string format_mermaid_graph(const Estimate& e) {
  std::ostringstream out;
  out << "xychart-beta\n";
  out << "  title \"KV memory growth for " << e.model_name << "\"\n";
  out << "  x-axis [\"reserved\", \"prefill\", \"decode\", \"budget\"]\n";
  out << "  y-axis \"GiB\" 0 --> " << std::max(1, static_cast<int>(std::ceil(bytes_to_gb(std::max(e.usable_memory_bytes, e.peak_working_bytes))))) << "\n";
  out << "  bar ["
      << fixed(bytes_to_gb(e.reserved_memory_bytes)) << ", "
      << fixed(bytes_to_gb(e.prefill_bytes)) << ", "
      << fixed(bytes_to_gb(e.decode_bytes)) << ", "
      << fixed(bytes_to_gb(e.usable_memory_bytes)) << "]\n";
  return out.str();
}

}  // namespace kvbench
