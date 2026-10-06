#pragma once

#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

namespace kvbench {
using Json = nlohmann::json;
class JsonError : public std::runtime_error {
 public:
  explicit JsonError(const std::string& message) : std::runtime_error(message) {}
};
/// Strict JSON: reject duplicate keys, malformed UTF-8, non-finite numbers,
/// inputs above 16 MiB and nesting deeper than 128. Integer storage is exact.
Json parse_json(const std::string& source);
/// Escape one JSON string, excluding its surrounding quotes.
std::string escape_json_string(const std::string& input);
}
