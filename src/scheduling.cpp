#include "kvbench/scheduling.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace kvbench {
namespace {
enum class Phase : std::uint8_t { waiting, prefill, decode, done };
struct Runtime {
  Phase phase = Phase::waiting;
  Count prefetched = 0;
  Count decoded = 0;
  Count blocks = 0;
  Count redo_remaining = 0;
  bool admitted_before = false;
};
LatencySummary summarize(std::vector<double> values) {
  LatencySummary s;
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  const auto rank = [&](double p) {
    const auto index =
        static_cast<std::size_t>(std::ceil(p * static_cast<double>(values.size()))) - 1;
    return values.at(index);
  };
  s.p50 = rank(0.5);
  s.p90 = rank(0.9);
  s.p99 = rank(0.99);
  s.maximum = values.back();
  s.samples = static_cast<Count>(values.size());
  return s;
}
}  // namespace

std::string completion_name(CompletionStatus status) {
  switch (status) {
    case CompletionStatus::completed:
      return "completed";
    case CompletionStatus::truncated_time:
      return "truncated_time";
    case CompletionStatus::truncated_steps:
      return "truncated_steps";
    case CompletionStatus::infeasible:
      return "infeasible";
  }
  throw std::invalid_argument("invalid completion status");
}

ScheduleResult schedule(const PlannerConfig& c, const EventSink& sink) {
  validate_planner_config(c);
  const auto budget = estimate_budget(c);
  const auto& inputs = c.workload.requests;
  ScheduleResult result;
  std::vector<Runtime> states(inputs.size());
  for (const auto& request : inputs)
    result.requests.push_back({request.id, request.arrival_ms, {}, {}, {}, 0, false});
  const auto finish = [&]() {
    std::vector<double> ttft, latency;
    for (const auto& metrics : result.requests) {
      if (metrics.ttft_ms) ttft.push_back(*metrics.ttft_ms);
      if (metrics.latency_ms) latency.push_back(*metrics.latency_ms);
    }
    result.ttft = summarize(std::move(ttft));
    result.latency = summarize(std::move(latency));
    if (result.elapsed_ms > 0)
      result.tokens_per_second =
          static_cast<double>(result.generated_tokens) * 1000 / result.elapsed_ms;
    return result;
  };
  if (inputs.empty()) return finish();
  const bool impossible = std::any_of(inputs.begin(), inputs.end(), [&](const auto& request) {
    return sequence_pool_blocks(c.cache, checked_add(request.input_tokens, request.output_tokens)) >
               budget.num_blocks ||
           (!c.engine.enable_chunked_prefill &&
            request.input_tokens > c.engine.max_num_batched_tokens);
  });
  if (impossible) {
    result.status = CompletionStatus::infeasible;
    return finish();
  }
  std::vector<std::size_t> arrivals(inputs.size());
  std::iota(arrivals.begin(), arrivals.end(), std::size_t{0});
  std::stable_sort(arrivals.begin(), arrivals.end(),
                   [&](auto a, auto b) { return inputs[a].arrival_ms < inputs[b].arrival_ms; });
  std::size_t next_arrival = 0;
  std::deque<std::size_t> waiting;
  std::vector<std::size_t> running;  // Admission order; back is latest admitted.
  Count allocated = 0;
  double now = 0;
  while (result.completed_requests < inputs.size()) {
    if (result.steps == c.engine.max_steps) {
      result.status = CompletionStatus::truncated_steps;
      break;
    }
    if (running.empty() && waiting.empty() && next_arrival < arrivals.size())
      now = std::max(now, inputs[arrivals[next_arrival]].arrival_ms);
    if (now >= c.engine.max_simulated_time_ms) {
      now = c.engine.max_simulated_time_ms;
      result.status = CompletionStatus::truncated_time;
      break;
    }
    while (next_arrival < arrivals.size() && inputs[arrivals[next_arrival]].arrival_ms <= now)
      waiting.push_back(arrivals[next_arrival++]);
    Count remaining = c.engine.max_num_batched_tokens;
    Count prefill_tokens = 0;
    Count decode_seqs = 0;
    std::vector<bool> planned_decode(inputs.size(), false);
    std::vector<Count> planned_prefill(inputs.size(), 0);
    const auto decode_order = running;
    for (const auto index : decode_order) {
      auto& state = states[index];
      if (state.phase != Phase::decode || remaining == 0) continue;
      const Count target = checked_add(checked_add(inputs[index].input_tokens, state.decoded), 1);
      const Count needed = sequence_pool_blocks(c.cache, target);
      bool preempted_self = false;
      while (needed - state.blocks > budget.num_blocks - allocated) {
        if (running.empty()) throw std::logic_error("preemption invariant: no victim");
        const auto victim = running.back();
        // Admission-order decode scheduling means later victims have not yet
        // scheduled a token in this step. Never discard committed output.
        if (planned_decode[victim]) throw std::logic_error("preemption targeted scheduled decode");
        auto& old = states[victim];
        old.redo_remaining = old.phase == Phase::decode
                                 ? checked_add(inputs[victim].input_tokens, old.decoded)
                                 : old.prefetched;
        allocated = checked_sub(allocated, old.blocks);
        old.blocks = 0;
        old.prefetched = 0;
        old.phase = Phase::waiting;
        running.pop_back();
        waiting.push_front(victim);
        result.preemptions = checked_add(result.preemptions, 1);
        if (victim == index) {
          preempted_self = true;
          break;
        }
      }
      if (preempted_self) continue;
      allocated = checked_add(allocated, checked_sub(needed, state.blocks));
      state.blocks = needed;
      state.decoded = checked_add(state.decoded, 1);
      planned_decode[index] = true;
      remaining = checked_sub(remaining, 1);
      decode_seqs = checked_add(decode_seqs, 1);
    }
    const auto advance_prefill = [&](std::size_t index) {
      auto& state = states[index];
      const Count target = checked_add(inputs[index].input_tokens, state.decoded);
      Count advance = checked_sub(target, state.prefetched);
      if (c.engine.enable_chunked_prefill)
        advance = std::min(advance, remaining);
      else if (advance > remaining)
        return false;
      if (advance == 0) return false;
      const auto fits = [&](Count amount) {
        const Count blocks = sequence_pool_blocks(c.cache, checked_add(state.prefetched, amount));
        return checked_sub(blocks, state.blocks) <= checked_sub(budget.num_blocks, allocated);
      };
      if (!fits(advance)) {
        if (!c.engine.enable_chunked_prefill) return false;
        Count low = 0, high = advance;
        while (low < high) {
          const Count mid = checked_add(low, ceil_div(high - low, 2));
          if (fits(mid))
            low = mid;
          else
            high = mid - 1;
        }
        advance = low;
        if (advance == 0) return false;
      }
      const Count blocks = sequence_pool_blocks(c.cache, checked_add(state.prefetched, advance));
      allocated = checked_add(allocated, checked_sub(blocks, state.blocks));
      state.blocks = blocks;
      planned_prefill[index] = advance;
      prefill_tokens = checked_add(prefill_tokens, advance);
      remaining = checked_sub(remaining, advance);
      return true;
    };
    // Existing prefills retain admission priority before FCFS waiting requests.
    bool prefill_blocked = false;
    for (const auto index : running) {
      if (states[index].phase == Phase::prefill && remaining > 0) {
        if (!advance_prefill(index)) {
          prefill_blocked = true;
          break;
        }
      }
    }
    while (!prefill_blocked && !waiting.empty() && remaining > 0 &&
           running.size() < c.engine.max_num_seqs) {
      const auto index = waiting.front();
      if (!advance_prefill(index)) break;
      waiting.pop_front();
      running.push_back(index);
      states[index].phase = Phase::prefill;
    }
    if (prefill_tokens == 0 && decode_seqs == 0) {
      result.status = CompletionStatus::infeasible;
      break;
    }
    const long double duration =
        c.engine.base_step_ms +
        static_cast<long double>(prefill_tokens) * c.engine.prefill_ms_per_token +
        static_cast<long double>(decode_seqs) * c.engine.decode_ms_per_seq;
    const long double end = static_cast<long double>(now) + duration;
    if (!std::isfinite(end) || end > std::numeric_limits<double>::max())
      throw std::overflow_error("modeled simulation time exceeds numeric range");
    const double finished = static_cast<double>(end);
    if (finished <= now) throw std::overflow_error("modeled step is below timestamp resolution");
    if (finished > c.engine.max_simulated_time_ms) {
      now = c.engine.max_simulated_time_ms;
      result.status = CompletionStatus::truncated_time;
      break;  // Do not commit token progress from a step that cannot complete.
    }
    if (allocated > budget.num_blocks || running.size() > c.engine.max_num_seqs ||
        checked_add(prefill_tokens, decode_seqs) > c.engine.max_num_batched_tokens)
      throw std::logic_error("scheduler resource invariant violated");
    const Count sum = std::accumulate(
        running.begin(), running.end(), Count{0},
        [&](Count total, std::size_t index) { return checked_add(total, states[index].blocks); });
    if (sum != allocated) throw std::logic_error("scheduler allocation accounting diverged");
    result.peak_blocks = std::max(result.peak_blocks, allocated);
    if (allocated == budget.num_blocks && !result.saturation_time_ms)
      result.saturation_time_ms = now;
    if (sink)
      sink({result.steps, now, finished, static_cast<Count>(running.size()), allocated,
            prefill_tokens, decode_seqs});
    for (const auto index : running) {
      auto& state = states[index];
      auto& metrics = result.requests[index];
      if (!state.admitted_before) {
        metrics.queueing_delay_ms = now - inputs[index].arrival_ms;
        state.admitted_before = true;
      }
      if (planned_prefill[index] != 0) {
        const Count redo = std::min(state.redo_remaining, planned_prefill[index]);
        result.recomputed_tokens = checked_add(result.recomputed_tokens, redo);
        state.redo_remaining = checked_sub(state.redo_remaining, redo);
        state.prefetched = checked_add(state.prefetched, planned_prefill[index]);
        if (state.prefetched == checked_add(inputs[index].input_tokens, state.decoded))
          state.phase = Phase::decode;
      }
      if (planned_decode[index]) {
        result.generated_tokens = checked_add(result.generated_tokens, 1);
        metrics.output_tokens = state.decoded;
        if (!metrics.ttft_ms) metrics.ttft_ms = finished - inputs[index].arrival_ms;
      }
      if (state.phase == Phase::decode && state.decoded == inputs[index].output_tokens) {
        metrics.completed = true;
        metrics.latency_ms = finished - inputs[index].arrival_ms;
        state.phase = Phase::done;
        result.completed_requests = checked_add(result.completed_requests, 1);
        allocated = checked_sub(allocated, state.blocks);
        state.blocks = 0;
      }
    }
    std::erase_if(running, [&](auto index) { return states[index].phase == Phase::done; });
    now = finished;
    result.steps = checked_add(result.steps, 1);
  }
  result.elapsed_ms = now;
  return finish();
}
}  // namespace kvbench
