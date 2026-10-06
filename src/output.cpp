#include "kvbench/output.hpp"
#include <sstream>
#include <stdexcept>

namespace kvbench {
namespace {
Json optional_time(const std::optional<double>& value) { return value ? Json(*value) : Json(nullptr); }
Json latency(const LatencySummary& s) { return {{"p50_ms",s.p50},{"p90_ms",s.p90},{"p99_ms",s.p99},{"max_ms",s.maximum},{"samples",s.samples}}; }
std::string markdown(const std::string& value) {
  std::string out;
  for (const char c : value) {
    if (c == '|') out += "\\|";
    else if (c == '\n' || c == '\r') out += "<br>";
    else if (c == '<') out += "&lt;";
    else if (c == '>') out += "&gt;";
    else if (c == '&') out += "&amp;";
    else if (c == '\\') out += "\\\\";
    else out += c;
  }
  return out;
}
std::string csv(const std::string& value) {
  std::string out = "\"";
  for (const char c : value) { if (c == '"') out += "\"\""; else out += c; }
  return out + "\"";
}
void flatten(const Json& value, const std::string& path, std::vector<std::pair<std::string,std::string>>& rows) {
  if (value.is_object()) for (const auto& [key, item] : value.items()) flatten(item, path.empty() ? key : path + "." + key, rows);
  else rows.emplace_back(path, value.is_string() ? value.get<std::string>() : value.dump());
}
}
Json budget_report(const PlannerConfig& c, const BudgetResult& b, const std::string& command) {
  return {{"kvbench_result",2},{"command",command},{"model_name",c.model_name},{"measurement",false},{"time_model_calibrated",c.engine.calibrated},{"warnings",c.warnings},
    {"budget",{{"total_gpu_bytes",b.total_gpu_bytes},{"utilization_bytes",b.utilization_bytes},{"weights_bytes",b.weights.bytes},{"weights_estimated",b.weights.estimated},{"non_torch_overhead_bytes",b.non_torch_overhead_bytes},{"activation_peak_bytes",b.activation_peak_bytes},{"cudagraph_memory_bytes",b.cudagraph_memory_bytes},{"overhead_deficit_bytes",b.overhead_deficit_bytes},{"kv_pool_bytes",b.kv_pool_bytes},{"kv_data_bytes_per_token",b.cache.data_bytes_per_token},{"bytes_per_block",b.cache.bytes_per_block},{"metadata_bytes_per_block",b.cache.metadata_bytes_per_block},{"constant_metadata_bytes",b.cache.constant_metadata_bytes},{"num_blocks",b.num_blocks},{"max_tokens_in_cache",b.max_tokens_in_cache},{"blocks_per_sequence",b.blocks_per_sequence},{"max_concurrent_sequences",b.max_concurrent_sequences}}},
    {"static_worst_case",{{"context_tokens",c.workload.context_tokens},{"decode_tokens",c.workload.decode_tokens},{"concurrent_requests",c.workload.concurrent_requests},{"prefill_cache_bytes",b.workload.prefill_cache_bytes},{"decode_cache_bytes",b.workload.decode_cache_bytes},{"prefill_bytes",b.workload.prefill_bytes},{"decode_bytes",b.workload.decode_bytes},{"peak_bytes",b.workload.peak_bytes},{"available_bytes",b.workload.available_bytes},{"fits",b.workload.fits},{"risk",b.workload.risk},{"max_concurrency",b.workload.max_concurrency},{"max_concurrency_is_lower_bound",b.workload.concurrency_bounded}}}};
}
Json schedule_report(const PlannerConfig& c, const BudgetResult& b, const ScheduleResult& r) {
  Json j = budget_report(c,b,"schedule");
  j["schedule"] = {{"status",completion_name(r.status)},{"steps",r.steps},{"peak_blocks",r.peak_blocks},{"peak_cache_bytes",checked_add(checked_mul(r.peak_blocks,b.cache.bytes_per_block),b.cache.constant_metadata_bytes)},{"preemptions",r.preemptions},{"recomputed_tokens",r.recomputed_tokens},{"generated_tokens",r.generated_tokens},{"completed_requests",r.completed_requests},{"elapsed_ms",r.elapsed_ms},{"tokens_per_second",r.tokens_per_second},{"saturation_time_ms",optional_time(r.saturation_time_ms)},{"ttft",latency(r.ttft)},{"latency",latency(r.latency)},{"requests",Json::array()}};
  for (const auto& m : r.requests) j["schedule"]["requests"].push_back({{"id",m.id},{"arrival_ms",m.arrival_ms},{"queueing_delay_ms",optional_time(m.queueing_delay_ms)},{"ttft_ms",optional_time(m.ttft_ms)},{"latency_ms",optional_time(m.latency_ms)},{"output_tokens",m.output_tokens},{"completed",m.completed}});
  return j;
}
Json event_report(const ScheduleStep& s) { return {{"kvbench_event",2},{"step",s.step},{"started_ms",s.started_ms},{"finished_ms",s.finished_ms},{"running",s.running},{"allocated_blocks",s.allocated_blocks},{"prefill_tokens",s.prefill_tokens},{"decode_seqs",s.decode_seqs}}; }
std::string render_report(const Json& report, const std::string& format) {
  if (format == "json") return report.dump(2) + "\n";
  std::vector<std::pair<std::string,std::string>> rows;
  flatten(report,"",rows);
  std::ostringstream out;
  if (format == "markdown") out << "Model output, not a hardware measurement; time model uncalibrated unless stated.\n\n| Metric | Value |\n| --- | --- |\n";
  else if (format == "csv") out << "metric,value\n";
  else if (format == "text") out << "Model output, not a hardware measurement; time model uncalibrated unless stated.\n";
  else throw std::invalid_argument("unsupported output format");
  for (const auto& [key,value] : rows) {
    if (format == "markdown") out << "| " << markdown(key) << " | " << markdown(value) << " |\n";
    else if (format == "csv") out << csv(key) << ',' << csv(value) << '\n';
    else out << key << ": " << Json(value).dump() << '\n';
  }
  return out.str();
}
std::string escape_mermaid(const std::string& label) {
  std::string escaped;
  for (const char raw : label) {
    const auto c = static_cast<unsigned char>(raw);
    if (c == '"' || c == '[' || c == ']' || c == '|' || c == '\\' || c == '<' || c == '>' || c == '&' || c < 32 || c == 127) escaped += "&#" + std::to_string(c) + ";";
    else escaped += static_cast<char>(c);
  }
  return escaped;
}
std::string budget_graph(const PlannerConfig& c, const BudgetResult& b) {
  std::ostringstream out;
  out << "flowchart LR\n  model[\"" << escape_mermaid(c.model_name) << "\"] --> weights[\"Weights: " << b.weights.bytes << " bytes\"]\n  gpu[\"GPU: " << b.total_gpu_bytes << " bytes\"] --> pool[\"KV pool: " << b.kv_pool_bytes << " bytes\"]\n  pool --> blocks[\"" << b.num_blocks << " blocks\"]\n";
  return out.str();
}
}
