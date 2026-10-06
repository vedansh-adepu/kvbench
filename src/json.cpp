#include "kvbench/json.hpp"

#include <set>
#include <vector>

namespace kvbench {
Json parse_json(const std::string& source) {
  if (source.size() > 16 * 1024 * 1024) throw JsonError("JSON input exceeds 16 MiB");
  std::vector<std::set<std::string>> objects;
  const auto callback = [&objects](int depth, Json::parse_event_t event, Json& parsed) {
    if (depth > 128) throw JsonError("JSON nesting exceeds 128");
    if (event == Json::parse_event_t::object_start) objects.emplace_back();
    if (event == Json::parse_event_t::key) {
      if (!objects.back().insert(parsed.get<std::string>()).second) throw JsonError("duplicate JSON object key");
    }
    if (event == Json::parse_event_t::object_end) objects.pop_back();
    return true;
  };
  try {
    return Json::parse(source, callback, true, false);
  } catch (const Json::exception&) {
    throw JsonError("invalid JSON syntax, encoding or numeric range");
  }
}
std::string escape_json_string(const std::string& input) {
  try {
    const std::string quoted = Json(input).dump();
    return quoted.substr(1, quoted.size() - 2);
  } catch (const Json::exception&) {
    throw JsonError("invalid UTF-8 in output label");
  }
}
}
