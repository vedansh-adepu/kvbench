#pragma once

#include <optional>
#include <string>
#include <vector>

namespace kvbench {

enum class DType {
  fp32,
  fp16,
  bf16,
  int8,
  int4,
};

struct ModelConfig {
  std::string name = "unnamed-model";
  int layers = 0;
  int attention_heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int hidden_size = 0;
  DType dtype = DType::fp16;
};

struct WorkloadConfig {
  int context_tokens = 0;
  int decode_tokens = 0;
  int concurrent_requests = 1;
  int batch_size = 1;
  int prefill_chunk_size = 1024;
};

struct SystemConfig {
  double gpu_memory_gb = 24.0;
  double reserved_memory_gb = 3.0;
  DType kv_quantization = DType::fp16;
  int page_size_tokens = 16;
};

struct RequestConfig {
  std::string id;
  int arrival_ms = 0;
  int input_tokens = 0;
  int output_tokens = 0;
};

struct Config {
  ModelConfig model;
  WorkloadConfig workload;
  SystemConfig system;
  std::vector<RequestConfig> requests;
};

std::string dtype_name(DType dtype);
double dtype_bytes(DType dtype);
DType parse_dtype(const std::string& name);

Config load_config_file(const std::string& path);
Config parse_config_text(const std::string& source, const std::string& source_name);

void validate_config(const Config& config, const std::string& source_name);

}  // namespace kvbench
