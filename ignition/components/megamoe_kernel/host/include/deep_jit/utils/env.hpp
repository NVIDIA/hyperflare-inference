#pragma once

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <locale>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <deep_jit/utils/exception.hpp>

namespace deep_jit {

// Parse a whole string as an integer; malformed or partial values throw.
template <typename T>
inline T parse_integer(const char* raw, const std::string& name) {
    T value;
    const char* end = raw + std::strlen(raw);
    const auto [position, error] = std::from_chars(raw, end, value);
    if (error != std::errc{} or position != end)
        DJ_PANIC("invalid value for {}: {}", name, raw);
    return value;
}

template <typename T>
inline T get_env(const std::string& name, const T& default_value = T()) {
    const char* raw = std::getenv(name.c_str());
    if (raw == nullptr)
        return default_value;
    if constexpr (std::is_same_v<T, std::string>) {
        return std::string(raw);
    } else if constexpr (std::is_same_v<T, bool>) {
        // Accept true/false, yes/no, and integer (non-zero is true)
        // notations, case-insensitively; anything else is an error.
        std::string value(raw);
        for (char& c : value)
            c = std::tolower(c, std::locale::classic());
        if (value == "true" or value == "yes")
            return true;
        if (value == "false" or value == "no")
            return false;
        return parse_integer<long long>(raw, name) != 0;
    } else if constexpr (std::is_integral_v<T>) {
        return parse_integer<T>(raw, name);
    } else if constexpr (std::is_enum_v<T>) {
        return static_cast<T>(parse_integer<std::underlying_type_t<T>>(raw, name));
    } else {
        DJ_STATIC_ASSERT((std::is_same_v<T, void>), "unsupported type for get_env");
        return T();
    }
}

}  // namespace deep_jit
