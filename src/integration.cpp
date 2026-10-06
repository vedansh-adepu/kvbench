#include "kvbench/integration.hpp"

#include <cmath>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace kvbench {
std::string read_text_input(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw JsonError("cannot open input file");
  std::string value;
  char chunk[4096];
  while (file.read(chunk, sizeof(chunk)) || file.gcount() != 0) {
    value.append(chunk, static_cast<std::size_t>(file.gcount()));
    if (value.size() > std::size_t{16} * 1024 * 1024) throw JsonError("input exceeds 16 MiB");
  }
  if (file.bad()) throw JsonError("cannot read input file");
  return value;
}

Json import_hf(const std::string& source) {
  const Json hf = parse_json(source);
  if (!hf.is_object()) throw JsonError("HF config must be an object");
  Json config = {{"schema_version", 2}, {"model", Json::object()}, {"weights", Json::object()}};
  auto& model = config["model"];
  auto& weights = config["weights"];
  std::set<std::string> mapped;
  const auto copy = [&](const std::string& from, Json& target, const std::string& to) {
    if (hf.contains(from)) {
      target[to] = hf.at(from);
      mapped.insert(from);
    }
  };
  copy("num_hidden_layers", model, "num_layers");
  copy("num_attention_heads", model, "num_attention_heads");
  copy("num_key_value_heads", model, "num_kv_heads");
  copy("hidden_size", model, "hidden_size");
  if (hf.contains("head_dim") && !hf.at("head_dim").is_null()) copy("head_dim", model, "head_dim");
  copy("intermediate_size", weights, "intermediate_size");
  copy("vocab_size", weights, "vocab_size");
  copy("tie_word_embeddings", weights, "tie_word_embeddings");
  copy("num_local_experts", weights, "num_experts");
  if (hf.contains("num_experts") && hf.contains("num_local_experts"))
    throw JsonError("ambiguous HF expert count");
  copy("num_experts", weights, "num_experts");
  copy("num_shared_experts", weights, "num_shared_experts");
  if (hf.contains("kv_lora_rank") || hf.contains("qk_rope_head_dim")) {
    model["attention_type"] = "mla";
    copy("kv_lora_rank", model, "kv_lora_rank");
    copy("qk_rope_head_dim", model, "qk_rope_head_dim");
  }
  bool sliding_enabled = true;
  if (hf.contains("use_sliding_window")) {
    if (!hf.at("use_sliding_window").is_boolean())
      throw JsonError("use_sliding_window must be boolean");
    sliding_enabled = hf.at("use_sliding_window").get<bool>();
    mapped.insert("use_sliding_window");
  }
  if (sliding_enabled && hf.contains("sliding_window") && !hf.at("sliding_window").is_null()) {
    if (model.value("attention_type", std::string{}) == "mla")
      throw JsonError("HF sliding MLA requires explicit layer cache geometry");
    copy("sliding_window", model, "sliding_window");
    if (!hf.contains("layer_types")) model["attention_type"] = "sliding";
  }
  if (hf.contains("layer_types")) {
    if (!hf.at("layer_types").is_array()) throw JsonError("HF layer_types must be an array");
    model["layer_types"] = Json::array();
    for (const auto& type : hf.at("layer_types")) {
      if (!type.is_string()) throw JsonError("HF layer_types entries must be strings");
      const auto name = type.get<std::string>();
      if (name == "full_attention" || name == "full")
        model["layer_types"].push_back("full");
      else if (name == "sliding_attention" || name == "sliding") {
        if (!sliding_enabled)
          throw JsonError("sliding layer conflicts with use_sliding_window=false");
        model["layer_types"].push_back("sliding");
      } else if (name == "mla")
        model["layer_types"].push_back("mla");
      else
        throw JsonError("unsupported HF layer type");
    }
    mapped.insert("layer_types");
  }
  const auto parsed = parse_planner_config(config.dump());
  std::vector<std::string> unmapped;
  for (const auto& [key, value] : hf.items()) {
    static_cast<void>(value);
    if (!mapped.contains(key)) unmapped.push_back(key);
  }
  auto warnings = parsed.warnings;
  warnings.emplace_back(
      "HF import supplies architecture only; inspect estimated weights and runtime defaults before "
      "planning");
  if (model.value("attention_type", std::string{}) == "mla")
    warnings.emplace_back("MLA requires explicit checkpoint weights before budget/simulation");
  return {{"kvbench_result", 2}, {"command", "import-hf"},      {"measurement", false},
          {"config", config},    {"unmapped_fields", unmapped}, {"warnings", warnings}};
}

LogObservations parse_vllm_log(const std::string& source) {
  if (source.size() > std::size_t{16} * 1024 * 1024) throw JsonError("vLLM log exceeds 16 MiB");
  const std::string numeric = R"(([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?))";
  const std::vector<std::pair<std::string, std::string>> patterns = {
      {"total_gpu_bytes", "total_gpu_memory\\s*\\(" + numeric + "\\s*GiB\\)"},
      {"utilization_bytes", "=\\s*" + numeric + "\\s*GiB"},
      {"weights_bytes", "model weights take\\s+" + numeric + "\\s*GiB"},
      {"non_torch_overhead_bytes", "non_torch_memory takes\\s+" + numeric + "\\s*GiB"},
      {"activation_peak_bytes", "activation peak memory takes\\s+" + numeric + "\\s*GiB"},
      {"cudagraph_memory_bytes", "CUDA graph memory(?: takes|:)\\s*" + numeric + "\\s*GiB"},
      {"kv_pool_bytes", "reserved for KV Cache is\\s+" + numeric + "\\s*GiB"},
      {"kv_pool_bytes", "Available KV cache memory:\\s*" + numeric + "\\s*GiB"}};
  LogObservations observed;
  const auto put = [&](const std::string& term, Count count) {
    if (observed.terms.contains(term) && observed.terms.at(term) != count)
      throw JsonError("conflicting startup observations; select one worker/run");
    observed.terms[term] = count;
  };
  std::istringstream lines(source);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.size() > 16384) throw JsonError("vLLM log line exceeds 16384 bytes");
    for (const auto& [term, pattern] : patterns) {
      std::smatch match;
      if (std::regex_search(line, match, std::regex(pattern, std::regex::icase))) {
        const auto number = parse_json(match[1].str());
        if (!number.is_number()) throw JsonError("invalid startup memory value");
        put(term, gib_to_bytes(number.get<long double>()));
      }
    }
    std::smatch utilization;
    if (std::regex_search(
            line, utilization,
            std::regex("gpu_memory_utilization\\s*\\(" + numeric + "\\)", std::regex::icase))) {
      const auto value = parse_json(utilization[1].str()).get<double>();
      if (!std::isfinite(value) || value <= 0 || value > 1)
        throw JsonError("invalid observed utilization");
      if (observed.utilization && *observed.utilization != value)
        throw JsonError("conflicting observed utilization");
      observed.utilization = value;
    }
    for (const auto& [term, pattern] : std::vector<std::pair<std::string, std::string>>{
             {"num_blocks", R"((?:# GPU blocks:|num_gpu_blocks\s*=|GPU blocks:)\s*([0-9]+))"},
             {"max_tokens_in_cache", R"(GPU KV cache size:\s*([0-9][0-9,]*)\s*tokens)"}}) {
      std::smatch match;
      if (std::regex_search(line, match, std::regex(pattern, std::regex::icase))) {
        std::string value = match[1].str();
        if (!std::regex_match(value, std::regex(R"((?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+))")))
          throw JsonError("invalid comma grouping in capacity count");
        std::erase(value, ',');
        const auto count = parse_json(value);
        if (!count.is_number_unsigned() &&
            (!count.is_number_integer() || count.get<std::int64_t>() < 0))
          throw JsonError("invalid capacity count");
        put(term, count.get<Count>());
      }
    }
  }
  if (observed.terms.empty()) throw JsonError("no supported vLLM startup observations found");
  return observed;
}

Json compare_calibration(const PlannerConfig& c, const LogObservations& observed,
                         double tolerance) {
  if (!std::isfinite(tolerance) || tolerance < 0)
    throw JsonError("tolerance must be finite and nonnegative");
  if (observed.utilization && (!std::isfinite(*observed.utilization) ||
                               *observed.utilization <= 0 || *observed.utilization > 1))
    throw JsonError("invalid observed utilization");
  if (!observed.terms.contains("kv_pool_bytes") && !observed.terms.contains("num_blocks") &&
      !observed.terms.contains("max_tokens_in_cache"))
    throw JsonError("log must include a KV capacity observation");
  const auto b = estimate_budget(c);
  const std::map<std::string, Count> predicted = {
      {"total_gpu_bytes", b.total_gpu_bytes},
      {"utilization_bytes", b.utilization_bytes},
      {"weights_bytes", b.weights.bytes},
      {"non_torch_overhead_bytes", b.non_torch_overhead_bytes},
      {"activation_peak_bytes", b.activation_peak_bytes},
      {"cudagraph_memory_bytes", b.cudagraph_memory_bytes},
      {"kv_pool_bytes", b.kv_pool_bytes},
      {"num_blocks", b.num_blocks},
      {"max_tokens_in_cache", b.max_tokens_in_cache}};
  Json report = {{"kvbench_result", 2},
                 {"command", "calibrate"},
                 {"measurement", false},
                 {"provenance", "supplied log; provenance not verified"},
                 {"tolerance_percent", tolerance},
                 {"within_tolerance", true},
                 {"terms", Json::array()}};
  for (const auto& [term, value] : predicted) {
    Json row = {{"term", term},
                {"predicted", value},
                {"observed", nullptr},
                {"absolute_error", nullptr},
                {"percent_error", nullptr},
                {"compared", false},
                {"units", term == "num_blocks"            ? "blocks"
                          : term == "max_tokens_in_cache" ? "tokens"
                                                          : "bytes"}};
    if (observed.terms.contains(term)) {
      const Count actual = observed.terms.at(term);
      const Count error = value >= actual ? checked_sub(value, actual) : checked_sub(actual, value);
      const double percent = actual != 0
                                 ? static_cast<double>(static_cast<long double>(error) * 100 /
                                                       static_cast<long double>(actual))
                                 : 0;
      const bool fits = actual == 0 ? error == 0 : percent <= tolerance;
      row["observed"] = actual;
      row["absolute_error"] = error;
      row["compared"] = true;
      row["within_tolerance"] = fits;
      if (actual != 0 || error == 0) row["percent_error"] = percent;
      if (!fits) report["within_tolerance"] = false;
    }
    report["terms"].push_back(row);
  }
  if (observed.utilization) {
    const double error = std::abs(c.engine.gpu_memory_utilization - *observed.utilization);
    const double percent = error * 100 / *observed.utilization;
    report["terms"].push_back({{"term", "gpu_memory_utilization"},
                               {"predicted", c.engine.gpu_memory_utilization},
                               {"observed", *observed.utilization},
                               {"absolute_error", error},
                               {"percent_error", percent},
                               {"compared", true},
                               {"within_tolerance", percent <= tolerance},
                               {"units", "ratio"}});
    if (percent > tolerance) report["within_tolerance"] = false;
  }
  return report;
}
}  // namespace kvbench
