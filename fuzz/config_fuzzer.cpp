#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "kvbench/budget.hpp"
#include "kvbench/json.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  try {
    const auto config = kvbench::parse_planner_config(std::string(
        reinterpret_cast<const char*>(data),
        size));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): byte input view only.
    static_cast<void>(kvbench::estimate_budget(config));
  } catch (const kvbench::JsonError&) {
  } catch (const std::invalid_argument&) {
  } catch (const std::overflow_error&) {
  } catch (const std::underflow_error&) {
  }
  return 0;
}
