#include "kvbench/planner_config.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

#include "kvbench/json.hpp"

namespace kvbench {
namespace {
void object(const Json& value, const std::string& path) {
  if (!value.is_object()) throw JsonError(path + " must be an object");
}
void keys(const Json& value, const std::set<std::string>& allowed, const std::string& path,
          bool legacy, std::vector<std::string>& warnings) {
  object(value, path);
  for (const auto& [name, unused] : value.items()) {
    static_cast<void>(unused);
    if (!allowed.contains(name)) {
      std::string message = path;
      message.append(": unknown field ").append(name);
      if (!legacy) throw JsonError(message);
      warnings.push_back(message);
    }
  }
}
Count integer(const Json& value, const std::string& field) {
  if (!value.is_number_integer()) throw JsonError(field + " must be an integer");
  if (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)
    throw JsonError(field + " must be nonnegative");
  return value.get<Count>();
}
Count count(const Json& source, const std::string& field, Count fallback) {
  return source.contains(field) ? integer(source.at(field), field) : fallback;
}
double number(const Json& source, const std::string& field, double fallback) {
  if (!source.contains(field)) return fallback;
  const auto& value = source.at(field);
  if (!value.is_number()) throw JsonError(field + " must be numeric");
  const double result = value.get<double>();
  if (!std::isfinite(result)) throw JsonError(field + " must be finite");
  return result;
}
std::string text(const Json& source, const std::string& field, const std::string& fallback) {
  if (!source.contains(field)) return fallback;
  if (!source.at(field).is_string()) throw JsonError(field + " must be a string");
  return source.at(field).get<std::string>();
}
bool flag(const Json& source, const std::string& field, bool fallback) {
  if (!source.contains(field)) return fallback;
  if (!source.at(field).is_boolean()) throw JsonError(field + " must be boolean");
  return source.at(field).get<bool>();
}
Bytes memory(const Json& source, const std::string& prefix, Bytes fallback) {
  const auto bytes = prefix + "_bytes";
  const auto gib = prefix + "_gib";
  if (source.contains(bytes) && source.contains(gib))
    throw JsonError("choose only one of " + bytes + " and " + gib);
  if (source.contains(bytes)) return integer(source.at(bytes), bytes);
  if (source.contains(gib)) return gib_to_bytes(number(source, gib, 0));
  return fallback;
}
LayerKind layer_kind(const std::string& value) {
  if (value == "full" || value == "standard") return LayerKind::full;
  if (value == "sliding") return LayerKind::sliding;
  if (value == "mla") return LayerKind::mla;
  throw JsonError("unknown attention layer type: " + value);
}
void positive(Count value, Count max, const std::string& field) {
  if (value == 0 || value > max)
    throw JsonError(field + " must be positive and <= " + std::to_string(max));
}
}  // namespace

PlannerConfig parse_planner_config(const std::string& source) {
  const Json root = parse_json(source);
  object(root, "config");
  const bool legacy = !root.contains("schema_version");
  PlannerConfig c;
  if (!legacy && integer(root.at("schema_version"), "schema_version") != 2)
    throw JsonError("unsupported schema_version; expected 2");
  if (legacy)
    c.warnings.emplace_back(
        "schema v1 is deprecated; _gb fields meant GiB; migrate to schema_version 2");
  keys(root,
       legacy ? std::set<std::string>{"model", "workload", "system", "requests"}
              : std::set<std::string>{"schema_version", "model", "weights", "engine", "hardware",
                                      "workload"},
       "config", legacy, c.warnings);
  if (!root.contains("model")) throw JsonError("missing model section");
  const Json& m = root.at("model");
  keys(m,
       legacy ? std::set<std::string>{"name", "layers", "attention_heads", "kv_heads", "head_dim",
                                      "hidden_size", "dtype"}
              : std::set<std::string>{"name", "num_layers", "num_attention_heads", "num_kv_heads",
                                      "head_dim", "hidden_size", "attention_type", "kv_lora_rank",
                                      "qk_rope_head_dim", "sliding_window", "layer_types"},
       "model", legacy, c.warnings);
  auto& a = c.cache.model;
  c.model_name = text(m, "name", c.model_name);
  a.num_layers = count(m, legacy ? "layers" : "num_layers", 0);
  a.num_attention_heads = count(m, legacy ? "attention_heads" : "num_attention_heads", 0);
  a.num_kv_heads = count(m, legacy ? "kv_heads" : "num_kv_heads", a.num_attention_heads);
  a.hidden_size = count(m, "hidden_size", 0);
  a.head_dim = count(m, "head_dim", 0);
  if (m.contains("head_dim") && a.head_dim == 0) throw JsonError("head_dim must be positive");
  if (m.contains("hidden_size") && a.hidden_size == 0)
    throw JsonError("hidden_size must be positive");
  if (a.head_dim == 0) {
    if (a.num_attention_heads == 0 || a.hidden_size == 0 ||
        a.hidden_size % a.num_attention_heads != 0)
      throw JsonError("head_dim requires hidden_size divisible by num_attention_heads");
    a.head_dim = a.hidden_size / a.num_attention_heads;
  }
  if (a.hidden_size == 0) a.hidden_size = checked_mul(a.num_attention_heads, a.head_dim);
  a.attention_type = legacy ? LayerKind::full : layer_kind(text(m, "attention_type", "standard"));
  a.kv_lora_rank = legacy ? 0 : count(m, "kv_lora_rank", 0);
  a.qk_rope_head_dim = legacy ? 0 : count(m, "qk_rope_head_dim", 0);
  a.sliding_window = legacy ? 0 : count(m, "sliding_window", 0);
  if (!legacy && m.contains("layer_types")) {
    const auto& layers = m.at("layer_types");
    if (!layers.is_array() || layers.size() > 4096)
      throw JsonError("layer_types must be an array of at most 4096 entries");
    for (const auto& layer : layers) {
      if (layer.is_string())
        a.layers.push_back({layer_kind(layer.get<std::string>()), a.sliding_window});
      else {
        keys(layer, {"type", "window_tokens"}, "model.layer_types", legacy, c.warnings);
        a.layers.push_back({layer_kind(text(layer, "type", "full")),
                            count(layer, "window_tokens", a.sliding_window)});
      }
    }
  }
  c.weights.intermediate_size = checked_mul(a.hidden_size, 4);
  const Json empty = Json::object();
  const auto& w = !legacy && root.contains("weights") ? root.at("weights") : empty;
  keys(w,
       {"weights_bytes", "weights_gib", "dtype", "intermediate_size", "vocab_size",
        "tie_word_embeddings", "gated_mlp", "num_experts", "num_shared_experts",
        "quantization_overhead_bytes"},
       "weights", false, c.warnings);
  if (w.contains("weights_bytes") || w.contains("weights_gib"))
    c.weights.weights_bytes = memory(w, "weights", 0);
  c.weights.dtype =
      parse_cache_dtype(text(w, "dtype", "bf16") == "fp8" ? "fp8_e4m3" : text(w, "dtype", "bf16"));
  c.weights.intermediate_size = count(w, "intermediate_size", c.weights.intermediate_size);
  c.weights.vocab_size = count(w, "vocab_size", c.weights.vocab_size);
  c.weights.tie_word_embeddings = flag(w, "tie_word_embeddings", true);
  c.weights.gated_mlp = flag(w, "gated_mlp", true);
  c.weights.num_experts = count(w, "num_experts", 1);
  c.weights.num_shared_experts = count(w, "num_shared_experts", 0);
  c.weights.quantization_overhead_bytes = count(w, "quantization_overhead_bytes", 0);
  const auto& e = !legacy && root.contains("engine") ? root.at("engine") : empty;
  keys(e,
       {"gpu_memory_utilization",
        "non_torch_overhead_bytes",
        "non_torch_overhead_gib",
        "activation_peak_bytes",
        "activation_peak_gib",
        "cudagraph_memory_bytes",
        "cudagraph_memory_gib",
        "block_size",
        "swa_extra_blocks",
        "kv_dtype",
        "scale_mode",
        "scale_bytes",
        "zero_point_bytes",
        "group_size",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "num_blocks",
        "preemption_mode",
        "base_step_ms",
        "prefill_ms_per_token",
        "decode_ms_per_seq",
        "max_simulated_time_ms",
        "max_steps",
        "calibrated"},
       "engine", false, c.warnings);
  auto& engine = c.engine;
  engine.gpu_memory_utilization =
      number(e, "gpu_memory_utilization", engine.gpu_memory_utilization);
  engine.non_torch_overhead_bytes =
      memory(e, "non_torch_overhead", engine.non_torch_overhead_bytes);
  engine.activation_peak_bytes = memory(e, "activation_peak", engine.activation_peak_bytes);
  engine.cudagraph_memory_bytes = memory(e, "cudagraph_memory", engine.cudagraph_memory_bytes);
  engine.block_size = count(e, "block_size", engine.block_size);
  engine.swa_extra_blocks = count(e, "swa_extra_blocks", engine.swa_extra_blocks);
  engine.max_num_seqs = count(e, "max_num_seqs", engine.max_num_seqs);
  engine.max_num_batched_tokens = count(e, "max_num_batched_tokens", engine.max_num_batched_tokens);
  engine.enable_chunked_prefill = flag(e, "enable_chunked_prefill", true);
  if (e.contains("num_blocks")) engine.num_blocks = integer(e.at("num_blocks"), "num_blocks");
  engine.preemption_mode = text(e, "preemption_mode", "recompute");
  engine.base_step_ms = number(e, "base_step_ms", engine.base_step_ms);
  engine.prefill_ms_per_token = number(e, "prefill_ms_per_token", engine.prefill_ms_per_token);
  engine.decode_ms_per_seq = number(e, "decode_ms_per_seq", engine.decode_ms_per_seq);
  engine.max_simulated_time_ms = number(e, "max_simulated_time_ms", engine.max_simulated_time_ms);
  engine.max_steps = count(e, "max_steps", engine.max_steps);
  engine.calibrated = flag(e, "calibrated", false);
  c.cache.quantization.dtype = parse_cache_dtype(text(e, "kv_dtype", "fp16"));
  const auto mode = text(e, "scale_mode", "none");
  if (mode == "none")
    c.cache.quantization.scale_mode = ScaleMode::none;
  else if (mode == "per_tensor")
    c.cache.quantization.scale_mode = ScaleMode::per_tensor;
  else if (mode == "per_token_head")
    c.cache.quantization.scale_mode = ScaleMode::per_token_head;
  else if (mode == "group")
    c.cache.quantization.scale_mode = ScaleMode::group;
  else
    throw JsonError("unknown scale_mode");
  c.cache.quantization.scale_bytes = count(e, "scale_bytes", 4);
  c.cache.quantization.zero_point_bytes = count(e, "zero_point_bytes", 0);
  c.cache.quantization.group_size = count(e, "group_size", 64);
  const auto& h = !legacy && root.contains("hardware") ? root.at("hardware") : empty;
  keys(h, {"total_gpu_bytes", "total_gpu_gib"}, "hardware", false, c.warnings);
  c.total_gpu_bytes = memory(h, "total_gpu", c.total_gpu_bytes);
  const auto& workload = root.contains("workload") ? root.at("workload") : empty;
  keys(workload,
       legacy ? std::set<std::string>{"context_tokens", "decode_tokens", "concurrent_requests",
                                      "batch_size", "prefill_chunk_size"}
              : std::set<std::string>{"context_tokens", "decode_tokens", "concurrent_requests",
                                      "requests"},
       "workload", legacy, c.warnings);
  c.workload.context_tokens = count(workload, "context_tokens", c.workload.context_tokens);
  c.workload.decode_tokens = count(workload, "decode_tokens", 0);
  c.workload.concurrent_requests = count(workload, "concurrent_requests", 1);
  if (legacy) {
    const auto& system = root.contains("system") ? root.at("system") : empty;
    keys(system, {"gpu_memory_gb", "reserved_memory_gb", "kv_quantization", "page_size_tokens"},
         "system", true, c.warnings);
    engine.gpu_memory_utilization = 1;
    c.total_gpu_bytes = gib_to_bytes(number(system, "gpu_memory_gb", 24));
    engine.non_torch_overhead_bytes = gib_to_bytes(number(system, "reserved_memory_gb", 3));
    engine.cudagraph_memory_bytes = 0;
    c.weights.weights_bytes = 0;
    const auto dtype = parse_cache_dtype(text(m, "dtype", "fp16"));
    c.cache.quantization.dtype =
        parse_cache_dtype(text(system, "kv_quantization", cache_dtype_name(dtype)));
    engine.block_size = count(system, "page_size_tokens", 16);
    engine.max_num_seqs = c.workload.concurrent_requests;
    const auto batch = count(workload, "batch_size", engine.max_num_seqs);
    const auto chunk = count(workload, "prefill_chunk_size", 1024);
    positive(batch, 1'000'000, "batch_size");
    positive(chunk, 1'000'000'000, "prefill_chunk_size");
    engine.max_num_batched_tokens = checked_mul(batch, chunk);
    const auto elements = checked_mul(checked_mul(a.num_layers, a.hidden_size),
                                      checked_mul(std::min(c.workload.context_tokens, chunk),
                                                  std::min(batch, c.workload.concurrent_requests)));
    engine.activation_peak_bytes = ceil_div(checked_mul(elements, dtype_bits(dtype)), 32);
    c.warnings.emplace_back(
        "v1 reserved memory maps to non_torch overhead; weights=0; old scratch heuristic maps to "
        "UNCALIBRATED fixed activation_peak");
  }
  const Json* requests = legacy
                             ? (root.contains("requests") ? &root.at("requests") : nullptr)
                             : (workload.contains("requests") ? &workload.at("requests") : nullptr);
  if (requests) {
    if (!requests->is_array() || requests->size() > 1'000'000)
      throw JsonError("requests must be an array of at most 1000000 entries");
    for (const auto& r : *requests) {
      keys(r, {"id", "arrival_ms", "input_tokens", "output_tokens"}, "requests[]", legacy,
           c.warnings);
      c.workload.requests.push_back(
          {text(r, "id", "request-" + std::to_string(c.workload.requests.size())),
           number(r, "arrival_ms", 0), count(r, "input_tokens", 0), count(r, "output_tokens", 0)});
    }
  }
  c.cache.block_size = engine.block_size;
  c.cache.swa_extra_blocks = engine.swa_extra_blocks;
  validate_planner_config(c);
  for (const auto& warning : validate_geometry(c.cache)) c.warnings.push_back(warning);
  if (!engine.calibrated)
    c.warnings.emplace_back("memory overheads and time model are UNCALIBRATED planning inputs");
  return c;
}

PlannerConfig load_planner_config(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw JsonError("cannot open config file");
  std::string source;
  char chunk[4096];
  while (stream.read(chunk, sizeof(chunk)) || stream.gcount() != 0) {
    source.append(chunk, static_cast<std::size_t>(stream.gcount()));
    if (source.size() > std::size_t{16} * 1024 * 1024) throw JsonError("config exceeds 16 MiB");
  }
  if (stream.bad()) throw JsonError("cannot read config file");
  return parse_planner_config(source);
}

void validate_planner_config(const PlannerConfig& c) {
  if (c.schema_version != 2) throw JsonError("unsupported planner schema");
  static_cast<void>(cache_layout(c.cache));
  if (c.cache.block_size != c.engine.block_size ||
      c.cache.swa_extra_blocks != c.engine.swa_extra_blocks)
    throw JsonError("engine/cache geometry mismatch");
  if (c.total_gpu_bytes == 0) throw JsonError("total_gpu_bytes must be positive");
  if (!std::isfinite(c.engine.gpu_memory_utilization) || c.engine.gpu_memory_utilization <= 0 ||
      c.engine.gpu_memory_utilization > 1)
    throw JsonError("gpu_memory_utilization must be finite in (0,1]");
  positive(c.engine.max_num_seqs, 1'000'000, "max_num_seqs");
  positive(c.engine.max_num_batched_tokens, 1'000'000'000, "max_num_batched_tokens");
  positive(c.engine.max_steps, std::numeric_limits<Count>::max(), "max_steps");
  if (c.engine.preemption_mode != "recompute")
    throw JsonError("only recompute preemption is supported");
  for (const auto value :
       {c.engine.base_step_ms, c.engine.prefill_ms_per_token, c.engine.decode_ms_per_seq}) {
    if (!std::isfinite(value) || value < 0)
      throw JsonError("time coefficients must be finite and nonnegative");
  }
  if (c.engine.base_step_ms == 0 && c.engine.prefill_ms_per_token == 0 &&
      c.engine.decode_ms_per_seq == 0)
    throw JsonError("time model cannot have all zero coefficients");
  if (!std::isfinite(c.engine.max_simulated_time_ms) || c.engine.max_simulated_time_ms <= 0)
    throw JsonError("max_simulated_time_ms must be finite and positive");
  positive(c.workload.context_tokens, std::numeric_limits<Count>::max(), "context_tokens");
  positive(c.workload.concurrent_requests, std::numeric_limits<Count>::max(),
           "concurrent_requests");
  static_cast<void>(checked_add(c.workload.context_tokens, c.workload.decode_tokens));
  positive(c.weights.intermediate_size, 16777216, "intermediate_size");
  positive(c.weights.vocab_size, 16777216, "vocab_size");
  positive(c.weights.num_experts, 65536, "num_experts");
  if (c.weights.num_shared_experts > 65536) throw JsonError("num_shared_experts exceeds maximum");
  static_cast<void>(dtype_bits(c.weights.dtype));
  std::set<std::string> ids;
  if (c.workload.requests.size() > 1'000'000) throw JsonError("too many requests");
  for (const auto& r : c.workload.requests) {
    if (r.id.empty() || !ids.insert(r.id).second)
      throw JsonError("request ids must be nonempty and unique");
    if (!std::isfinite(r.arrival_ms) || r.arrival_ms < 0)
      throw JsonError("arrival_ms must be finite and nonnegative");
    if (r.input_tokens == 0) throw JsonError("input_tokens must be positive");
    static_cast<void>(checked_add(r.input_tokens, r.output_tokens));
  }
}
}  // namespace kvbench
