#include "kvbench/json.hpp"

#include <cctype>
#include <charconv>
#include <cstdlib>
#include <sstream>

namespace kvbench {
namespace {

class Parser {
 public:
  explicit Parser(const std::string& source) : source_(source) {}

  Json parse() {
    skip_ws();
    Json value = parse_value();
    skip_ws();
    if (!eof()) {
      fail("unexpected trailing input");
    }
    return value;
  }

 private:
  const std::string& source_;
  std::size_t pos_ = 0;

  bool eof() const { return pos_ >= source_.size(); }
  char peek() const { return eof() ? '\0' : source_[pos_]; }
  char take() { return eof() ? '\0' : source_[pos_++]; }

  void skip_ws() {
    while (!eof() && std::isspace(static_cast<unsigned char>(peek()))) {
      ++pos_;
    }
  }

  [[noreturn]] void fail(const std::string& message) const {
    std::ostringstream out;
    out << "JSON parse error at byte " << pos_ << ": " << message;
    throw JsonError(out.str());
  }

  void expect(char c) {
    if (take() != c) {
      std::string message = "expected '";
      message.push_back(c);
      message.push_back('\'');
      fail(message);
    }
  }

  bool consume_literal(const std::string& literal) {
    if (source_.compare(pos_, literal.size(), literal) == 0) {
      pos_ += literal.size();
      return true;
    }
    return false;
  }

  Json parse_value() {
    skip_ws();
    if (eof()) {
      fail("expected value");
    }
    const char c = peek();
    if (c == '{') return Json(parse_object());
    if (c == '[') return Json(parse_array());
    if (c == '"') return Json(parse_string());
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return Json(parse_number());
    if (consume_literal("true")) return Json(true);
    if (consume_literal("false")) return Json(false);
    if (consume_literal("null")) return Json(nullptr);
    fail("expected value");
  }

  Json::object parse_object() {
    Json::object object;
    expect('{');
    skip_ws();
    if (peek() == '}') {
      take();
      return object;
    }
    while (true) {
      skip_ws();
      if (peek() != '"') {
        fail("expected object key string");
      }
      std::string key = parse_string();
      skip_ws();
      expect(':');
      object.emplace(std::move(key), parse_value());
      skip_ws();
      const char c = take();
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}'");
    }
    return object;
  }

  Json::array parse_array() {
    Json::array array;
    expect('[');
    skip_ws();
    if (peek() == ']') {
      take();
      return array;
    }
    while (true) {
      array.push_back(parse_value());
      skip_ws();
      const char c = take();
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']'");
    }
    return array;
  }

  std::string parse_string() {
    std::string value;
    expect('"');
    while (!eof()) {
      const char c = take();
      if (c == '"') return value;
      if (static_cast<unsigned char>(c) < 0x20) {
        fail("control character in string");
      }
      if (c != '\\') {
        value.push_back(c);
        continue;
      }
      if (eof()) fail("unterminated escape");
      const char escape = take();
      switch (escape) {
        case '"': value.push_back('"'); break;
        case '\\': value.push_back('\\'); break;
        case '/': value.push_back('/'); break;
        case 'b': value.push_back('\b'); break;
        case 'f': value.push_back('\f'); break;
        case 'n': value.push_back('\n'); break;
        case 'r': value.push_back('\r'); break;
        case 't': value.push_back('\t'); break;
        case 'u':
          parse_unicode_escape(value);
          break;
        default:
          fail("invalid string escape");
      }
    }
    fail("unterminated string");
  }

  void parse_unicode_escape(std::string& value) {
    if (pos_ + 4 > source_.size()) {
      fail("short unicode escape");
    }
    unsigned code = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = take();
      code <<= 4;
      if (c >= '0' && c <= '9') code += static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') code += static_cast<unsigned>(10 + c - 'a');
      else if (c >= 'A' && c <= 'F') code += static_cast<unsigned>(10 + c - 'A');
      else fail("invalid unicode escape");
    }
    if (code <= 0x7F) {
      value.push_back(static_cast<char>(code));
    } else if (code <= 0x7FF) {
      value.push_back(static_cast<char>(0xC0 | (code >> 6)));
      value.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
      value.push_back(static_cast<char>(0xE0 | (code >> 12)));
      value.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      value.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
  }

  double parse_number() {
    const std::size_t start = pos_;
    if (peek() == '-') ++pos_;
    if (peek() == '0') {
      ++pos_;
    } else {
      if (!std::isdigit(static_cast<unsigned char>(peek()))) fail("invalid number");
      while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
    }
    if (peek() == '.') {
      ++pos_;
      if (!std::isdigit(static_cast<unsigned char>(peek()))) fail("invalid number");
      while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
    }
    if (peek() == 'e' || peek() == 'E') {
      ++pos_;
      if (peek() == '+' || peek() == '-') ++pos_;
      if (!std::isdigit(static_cast<unsigned char>(peek()))) fail("invalid exponent");
      while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
    }
    const std::string text = source_.substr(start, pos_ - start);
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size()) {
      fail("invalid number");
    }
    return value;
  }
};

}  // namespace

bool Json::as_bool() const {
  if (!is_bool()) throw JsonError("expected boolean");
  return std::get<bool>(value_);
}

double Json::as_number() const {
  if (!is_number()) throw JsonError("expected number");
  return std::get<double>(value_);
}

const std::string& Json::as_string() const {
  if (!is_string()) throw JsonError("expected string");
  return std::get<std::string>(value_);
}

const Json::array& Json::as_array() const {
  if (!is_array()) throw JsonError("expected array");
  return std::get<array>(value_);
}

const Json::object& Json::as_object() const {
  if (!is_object()) throw JsonError("expected object");
  return std::get<object>(value_);
}

const Json& Json::at(const std::string& key) const {
  const auto& object = as_object();
  const auto it = object.find(key);
  if (it == object.end()) {
    throw JsonError("missing key: " + key);
  }
  return it->second;
}

const Json* Json::find(const std::string& key) const {
  if (!is_object()) return nullptr;
  const auto& object = as_object();
  const auto it = object.find(key);
  return it == object.end() ? nullptr : &it->second;
}

Json parse_json(const std::string& source) {
  return Parser(source).parse();
}

std::string escape_json_string(const std::string& input) {
  std::string out;
  for (const char c : input) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += "\\u00";
          constexpr char hex[] = "0123456789abcdef";
          out.push_back(hex[(c >> 4) & 0x0F]);
          out.push_back(hex[c & 0x0F]);
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

}  // namespace kvbench
