#include "kvbench/config.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "kvbench/json.hpp"

namespace kvbench {
namespace {

int as_int(const Json& json, const std::string& key) {
  const double value = json.at(key).as_number();
  if (std::floor(value) != value) {
    throw std::runtime_error("field '" + key + "' must be an integer");
  }
  return static_cast<int>(value);
}

int optional_int(const Json& json, const std::string& key, int fallback) {
  const Json* value = json.find(key);
  return value ? static_cast<int>(value->as_number()) : fallback;
}

double optional_double(const Json& json, const std::string& key, double fallback) {
  const Json* value = json.find(key);
  return value ? value->as_number() : fallback;
}

std::string optional_string(const Json& json, const std::string& key, const std::string& fallback) {
  const Json* value = json.find(key);
  return value ? value->as_string() : fallback;
}

ModelConfig parse_model(const Json& root) {
  const Json& model = root.at("model");
  ModelConfig config;
  config.name = optional_string(model, "name", config.name);
  config.layers = as_int(model, "layers");
  config.attention_heads = as_int(model, "attention_heads");
  config.kv_heads = optional_int(model, "kv_heads", config.attention_heads);
  config.head_dim = as_int(model, "head_dim");
  config.hidden_size = optional_int(model, "hidden_size", config.attention_heads * config.head_dim);
  config.dtype = parse_dtype(optional_string(model, "dtype", "fp16"));
  return config;
}

WorkloadConfig parse_workload(const Json& root) {
  WorkloadConfig config;
  const Json* workload = root.find("workload");
  if (!workload) {
    return config;
  }
  config.context_tokens = optional_int(*workload, "context_tokens", config.context_tokens);
  config.decode_tokens = optional_int(*workload, "decode_tokens", config.decode_tokens);
  config.concurrent_requests = optional_int(*workload, "concurrent_requests", config.concurrent_requests);
  config.batch_size = optional_int(*workload, "batch_size", config.concurrent_requests);
  config.prefill_chunk_size = optional_int(*workload, "prefill_chunk_size", config.prefill_chunk_size);
  return config;
}

SystemConfig parse_system(const Json& root, DType model_dtype) {
  SystemConfig config;
  const Json* system = root.find("system");
  if (!system) {
    config.kv_quantization = model_dtype;
    return config;
  }
  config.gpu_memory_gb = optional_double(*system, "gpu_memory_gb", config.gpu_memory_gb);
  config.reserved_memory_gb = optional_double(*system, "reserved_memory_gb", config.reserved_memory_gb);
  config.kv_quantization = parse_dtype(optional_string(*system, "kv_quantization", dtype_name(model_dtype)));
  config.page_size_tokens = optional_int(*system, "page_size_tokens", config.page_size_tokens);
  return config;
}

std::vector<RequestConfig> parse_requests(const Json& root) {
  std::vector<RequestConfig> requests;
  const Json* array = root.find("requests");
  if (!array) {
    return requests;
  }
  int index = 1;
  for (const Json& item : array->as_array()) {
    RequestConfig request;
    request.id = optional_string(item, "id", "request-" + std::to_string(index));
    request.arrival_ms = optional_int(item, "arrival_ms", 0);
    request.input_tokens = as_int(item, "input_tokens");
    request.output_tokens = as_int(item, "output_tokens");
    requests.push_back(request);
    ++index;
  }
  return requests;
}

}  // namespace

std::string dtype_name(DType dtype) {
  switch (dtype) {
    case DType::fp32: return "fp32";
    case DType::fp16: return "fp16";
    case DType::bf16: return "bf16";
    case DType::int8: return "int8";
    case DType::int4: return "int4";
  }
  return "unknown";
}

double dtype_bytes(DType dtype) {
  switch (dtype) {
    case DType::fp32: return 4.0;
    case DType::fp16: return 2.0;
    case DType::bf16: return 2.0;
    case DType::int8: return 1.0;
    case DType::int4: return 0.5;
  }
  return 0.0;
}

DType parse_dtype(const std::string& name) {
  if (name == "fp32") return DType::fp32;
  if (name == "fp16") return DType::fp16;
  if (name == "bf16") return DType::bf16;
  if (name == "int8") return DType::int8;
  if (name == "int4") return DType::int4;
  throw std::runtime_error("unsupported dtype or quantization: " + name);
}

Config load_config_file(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("failed to open config: " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return parse_config_text(buffer.str(), path);
}

Config parse_config_text(const std::string& source, const std::string& source_name) {
  const Json root = parse_json(source);
  if (!root.is_object()) {
    throw std::runtime_error("config root must be a JSON object");
  }
  Config config;
  config.model = parse_model(root);
  config.workload = parse_workload(root);
  config.system = parse_system(root, config.model.dtype);
  config.requests = parse_requests(root);
  if (!config.requests.empty()) {
    int max_context = 0;
    int max_decode = 0;
    for (const auto& request : config.requests) {
      max_context = std::max(max_context, request.input_tokens);
      max_decode = std::max(max_decode, request.output_tokens);
    }
    config.workload.context_tokens = max_context;
    config.workload.decode_tokens = max_decode;
    config.workload.concurrent_requests = static_cast<int>(config.requests.size());
    config.workload.batch_size = static_cast<int>(config.requests.size());
  }
  validate_config(config, source_name);
  return config;
}

void validate_config(const Config& config, const std::string& source_name) {
  auto require_positive = [&](int value, const std::string& field) {
    if (value <= 0) {
      throw std::runtime_error(source_name + ": " + field + " must be positive");
    }
  };
  require_positive(config.model.layers, "model.layers");
  require_positive(config.model.attention_heads, "model.attention_heads");
  require_positive(config.model.kv_heads, "model.kv_heads");
  require_positive(config.model.head_dim, "model.head_dim");
  require_positive(config.model.hidden_size, "model.hidden_size");
  require_positive(config.workload.concurrent_requests, "workload.concurrent_requests");
  require_positive(config.workload.batch_size, "workload.batch_size");
  require_positive(config.workload.prefill_chunk_size, "workload.prefill_chunk_size");
  require_positive(config.system.page_size_tokens, "system.page_size_tokens");
  if (config.workload.context_tokens < 0 || config.workload.decode_tokens < 0) {
    throw std::runtime_error(source_name + ": token counts cannot be negative");
  }
  if (config.workload.context_tokens == 0 && config.requests.empty()) {
    throw std::runtime_error(source_name + ": workload.context_tokens must be positive");
  }
  if (config.system.gpu_memory_gb <= 0) {
    throw std::runtime_error(source_name + ": system.gpu_memory_gb must be positive");
  }
  if (config.system.reserved_memory_gb < 0 || config.system.reserved_memory_gb >= config.system.gpu_memory_gb) {
    throw std::runtime_error(source_name + ": system.reserved_memory_gb must be non-negative and below gpu_memory_gb");
  }
  for (const auto& request : config.requests) {
    if (request.arrival_ms < 0 || request.input_tokens <= 0 || request.output_tokens < 0) {
      throw std::runtime_error(source_name + ": request '" + request.id + "' has invalid scheduler fields");
    }
  }
}

}  // namespace kvbench
