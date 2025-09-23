// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "json.hpp"
#include <sstream>

namespace rocpd
{

/**
 * @brief Create a new json instance wrapped in a shared_ptr.
 *
 * Returns a std::shared_ptr managing a newly allocated empty json object.
 *
 * @return std::shared_ptr<json> Shared pointer owning the new json instance.
 */
std::shared_ptr<json>
json::create()
{
    return std::shared_ptr<json>(new json());
}

/**
 * @brief Insert or update a value by key in the JSON object.
 *
 * Stores a copy of the provided value in the object's internal map under
 * the given key. If an entry with the same key already exists, it is
 * replaced. The value is stored as a std::shared_ptr to a copy of the
 * provided json_value (shared ownership semantics).
 *
 * @param key Key under which to store the value.
 * @param value Value to store (copied and wrapped in a shared_ptr).
 */
void
json::set(const std::string& key, const json_value& value)
{
    data[key] = std::make_shared<json_value>(value);
}

/**
 * @brief Serialize the stored key-value pairs to a JSON-formatted string.
 *
 * Produces a JSON-like representation of this object's internal map in the
 * form `{"key1": value1, "key2": value2, ...}` by iterating over all entries
 * in `data` and serializing each value via `stringify()`. Keys are emitted
 * quoted; values are formatted according to their variant type (strings,
 * numbers, booleans, null, arrays, nested objects, etc.).
 *
 * Note: The output order of fields matches the iteration order of the
 * underlying `data` container. This method does not perform additional
 * escaping beyond what `stringify()` provides and does not throw on its own.
 *
 * @return std::string JSON-formatted representation of the object.
 */
std::string
json::to_string() const
{
    std::ostringstream oss;
    oss << "{";
    bool first = true;

    for(const auto& [key, value] : data)
    {
        if(!first) oss << ", ";
        first = false;

        oss << "\"" << key << "\": " << stringify(value);
    }

    oss << "}";
    return oss.str();
}

/**
 * @brief Serialize a json_value variant to a JSON-formatted string.
 *
 * Converts the held variant to its JSON textual representation. Supported types:
 * - std::string: emitted with surrounding double quotes.
 * - bool: emitted as "true" or "false".
 * - std::nullptr_t: emitted as "null".
 * - std::vector<json>: emitted as a JSON array; each element is serialized via json::to_string().
 * - std::shared_ptr<json>: emitted as the nested object's to_string() result.
 * - Other scalar types (e.g., int, double): emitted via operator<<.
 *
 * @param value Shared pointer to the value to serialize. Must be non-null.
 * @return std::string JSON representation of the provided value.
 */
std::string
json::stringify(const std::shared_ptr<json_value>& value)
{
    std::ostringstream oss;
    std::visit(
        [&oss](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr(std::is_same_v<T, std::string>)
                oss << "\"" << arg << "\"";
            else if constexpr(std::is_same_v<T, bool>)
                oss << (arg ? "true" : "false");
            else if constexpr(std::is_same_v<T, std::nullptr_t>)
                oss << "null";
            else if constexpr(std::is_same_v<T, std::vector<json>>)
            {
                oss << "[";
                bool first = true;
                for(const auto& item : arg)
                {
                    if(!first) oss << ", ";
                    first = false;
                    oss << item.to_string();
                }
                oss << "]";
            }
            else if constexpr(std::is_same_v<T, std::shared_ptr<json>>)
            {
                oss << arg->to_string();
            }
            else
            {
                // handle int + double
                oss << arg;
            }
        },
        *value);
    return oss.str();
}

}  // namespace rocpd
