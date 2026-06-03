#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace kvbench {

class JsonError : public std::runtime_error {
 public:
  explicit JsonError(const std::string& message) : std::runtime_error(message) {}
};

class Json {
 public:
  using array = std::vector<Json>;
  using object = std::map<std::string, Json>;
  using value = std::variant<std::nullptr_t, bool, double, std::string, array, object>;

  Json() : value_(nullptr) {}
  explicit Json(value v) : value_(std::move(v)) {}

  bool is_null() const { return std::holds_alternative<std::nullptr_t>(value_); }
  bool is_bool() const { return std::holds_alternative<bool>(value_); }
  bool is_number() const { return std::holds_alternative<double>(value_); }
  bool is_string() const { return std::holds_alternative<std::string>(value_); }
  bool is_array() const { return std::holds_alternative<array>(value_); }
  bool is_object() const { return std::holds_alternative<object>(value_); }

  bool as_bool() const;
  double as_number() const;
  const std::string& as_string() const;
  const array& as_array() const;
  const object& as_object() const;

  const Json& at(const std::string& key) const;
  const Json* find(const std::string& key) const;

 private:
  value value_;
};

Json parse_json(const std::string& source);
std::string escape_json_string(const std::string& input);

}  // namespace kvbench
