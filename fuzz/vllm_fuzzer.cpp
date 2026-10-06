#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "kvbench/integration.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  try {
    static_cast<void>(kvbench::parse_vllm_log(std::string(
        reinterpret_cast<const char*>(data),
        size)));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): byte input view only.
  } catch (const kvbench::JsonError&) {
  } catch (const std::invalid_argument&) {
  } catch (const std::overflow_error&) {
  }
  return 0;
}
