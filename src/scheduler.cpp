#include "kvbench/scheduler.hpp"

#include <algorithm>

#include "kvbench/estimator.hpp"

namespace kvbench {
namespace {

enum class Phase {
  waiting,
  prefill,
  decode,
  done,
};

struct RuntimeRequest {
  RequestConfig config;
  Phase phase = Phase::waiting;
  int prefetched_tokens = 0;
  int decoded_tokens = 0;
};

bool all_done(const std::vector<RuntimeRequest>& requests) {
  return std::all_of(requests.begin(), requests.end(), [](const RuntimeRequest& request) {
    return request.phase == Phase::done;
  });
}

}  // namespace

SchedulerResult simulate_scheduler(const Config& config) {
  SchedulerResult result;
  std::vector<RuntimeRequest> requests;
  requests.reserve(config.requests.size());
  for (const auto& request : config.requests) {
    requests.push_back({request});
  }

  if (requests.empty()) {
    return result;
  }

  const double per_token_kv_bytes = static_cast<double>(config.model.layers) *
                                    static_cast<double>(config.model.kv_heads) *
                                    static_cast<double>(config.model.head_dim) *
                                    2.0 *
                                    dtype_bytes(config.system.kv_quantization);
  const double usable_memory_bytes = gb_to_bytes(config.system.gpu_memory_gb - config.system.reserved_memory_gb);
  const int step_ms = 10;

  for (int time_ms = 0; time_ms <= 24 * 60 * 60 * 1000 && !all_done(requests); time_ms += step_ms) {
    int active_prefill = 0;
    int active_decode = 0;
    long long active_kv_tokens = 0;

    for (auto& request : requests) {
      if (request.phase == Phase::waiting && request.config.arrival_ms <= time_ms) {
        request.phase = Phase::prefill;
      }
      if (request.phase == Phase::prefill) {
        ++active_prefill;
        request.prefetched_tokens = std::min(request.config.input_tokens,
            request.prefetched_tokens + config.workload.prefill_chunk_size);
        if (request.prefetched_tokens >= request.config.input_tokens) {
          request.phase = Phase::decode;
        }
      } else if (request.phase == Phase::decode) {
        ++active_decode;
        if (request.decoded_tokens < request.config.output_tokens) {
          ++request.decoded_tokens;
        }
        if (request.decoded_tokens >= request.config.output_tokens) {
          request.phase = Phase::done;
        }
      }

      if (request.phase == Phase::prefill || request.phase == Phase::decode || request.phase == Phase::done) {
        if (request.phase != Phase::done) {
          active_kv_tokens += request.prefetched_tokens + request.decoded_tokens;
        }
      }
    }

    const int active_requests = active_prefill + active_decode;
    const double memory_bytes = static_cast<double>(active_kv_tokens) * per_token_kv_bytes;
    const bool oom = memory_bytes > usable_memory_bytes;
    result.events.push_back({time_ms, active_requests, active_kv_tokens, memory_bytes, oom});
    if (memory_bytes > result.peak_memory_bytes) {
      result.peak_memory_bytes = memory_bytes;
      result.peak_time_ms = time_ms;
    }
    result.oom = result.oom || oom;
  }

  return result;
}

}  // namespace kvbench
