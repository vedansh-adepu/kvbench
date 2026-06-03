#pragma once

#include <string>
#include <vector>

#include "kvbench/config.hpp"

namespace kvbench {

struct SchedulerEvent {
  int time_ms = 0;
  int active_requests = 0;
  long long active_kv_tokens = 0;
  double memory_bytes = 0.0;
  bool oom = false;
};

struct SchedulerResult {
  std::vector<SchedulerEvent> events;
  double peak_memory_bytes = 0.0;
  int peak_time_ms = 0;
  bool oom = false;
};

SchedulerResult simulate_scheduler(const Config& config);

}  // namespace kvbench
