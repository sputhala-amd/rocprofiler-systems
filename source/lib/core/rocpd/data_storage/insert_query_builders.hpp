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

#pragma once

#include "common/traits.hpp"

#include <sstream>
#include <string>
#include <type_traits>

namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{
namespace queries
{

namespace query_builders
{

struct query_value_builder
{
    /**
     * @brief Construct a query_value_builder using an existing string stream.
     *
     * Stores a reference to the provided std::stringstream that will be used to
     * accumulate the generated SQL-like query fragments.
     *
     * @param ss Stream used to append query fragments (must outlive this builder).
     */
    query_value_builder(std::stringstream& ss)
    : _ss{ ss }
    {}

    template <typename... Values>
    /**
     * @brief Append a parenthesized, comma-separated list of values to the internal query stream.
     *
     * Writes "( " followed by each provided value and a separator into the builder's internal
     * std::stringstream, then closes the list with " )". Value formatting is delegated to the
     * builder's internal process_value overloads: string literals are quoted, optional types
     * are rendered as their contained value or the literal NULL, and other types are inserted
     * directly via operator<<.
     *
     * The values are separated by ", " and a trailing space is added before the closing parenthesis.
     *
     * @tparam Values Variadic value types to be inserted (deduced).
     * @param values The values to emit into the VALUES list.
     * @return query_value_builder& Reference to this builder to allow fluent chaining.
     */
    query_value_builder& set_values(Values&&... values)
    {
        auto i = sizeof...(values);
        _ss << "( ";
        ((process_value(values) << (i-- > 1 ? ", " : " ")), ...);
        _ss << ")";
        return *this;
    }

    /**
 * @brief Returns the current accumulated query fragment.
 *
 * Captures and returns a copy of the internal stringstream's contents as a std::string.
 *
 * @return std::string Snapshot of the query text built so far.
 */
std::string get_query_string() { return _ss.str(); }

private:
    template <typename T>
    /**
     * @brief Appends a string literal value to the internal stream enclosed in double quotes.
     *
     * Enabled only when `T` is detected as a string literal type by `common::traits::is_string_literal`.
     *
     * @tparam T Type of the provided value (must be a string literal type).
     * @param value The string literal to append; it will be written as `"value"`.
     * @return std::stringstream& Reference to the internal stringstream after writing the quoted value.
     */
    std::enable_if_t<common::traits::is_string_literal<T>(), std::stringstream&>
    process_value(T& value)
    {
        _ss << "\"" << value << "\"";
        return _ss;
    }

    template <typename T>
    /**
     * @brief Appends an optional value to the internal stringstream, using "NULL" for empty optionals.
     *
     * Enabled only when T is an optional-like type. If value.has_value() is true, the contained
     * value is streamed to the internal std::stringstream; otherwise the literal "NULL" is streamed.
     *
     * @param value Optional value to serialize into the query fragment.
     * @return std::stringstream& Reference to the internal stringstream after writing.
     */
    std::enable_if_t<common::traits::is_optional_v<std::decay_t<T>>, std::stringstream&>
    process_value(T& value)
    {
        if(value.has_value())
        {
            _ss << value.value();
        }
        else
        {
            _ss << "NULL";
        }
        return _ss;
    }

    template <typename T>
    /**
     * @brief Append a non-string, non-optional value to the internal stringstream.
     *
     * Writes the given value directly to the stored std::stringstream using operator<<.
     *
     * @tparam T Type of the value; constrained to types that are neither string literals nor std::optional-like.
     * @param value The value to append (forwarded by reference).
     * @return std::stringstream& Reference to the internal stringstream after insertion.
     */
    std::enable_if_t<!common::traits::is_string_literal<T>() &&
                         !common::traits::is_optional_v<std::decay_t<T>>,
                     std::stringstream&>
    process_value(T& value)
    {
        _ss << value;
        return _ss;
    }

private:
    std::stringstream& _ss;
};

struct query_columns_builder
{
    /**
     * @brief Constructs a query_columns_builder bound to a stringstream.
     *
     * Initializes the builder to write column-list fragments into the provided
     * stream and constructs an internal query_value_builder that writes into the
     * same stream (enabling fluent "columns ... VALUES ( ... )" composition).
     *
     * @param ss Reference to the std::stringstream used to accumulate the query.
     */
    query_columns_builder(std::stringstream& ss)
    : _ss{ ss }
    , _query_value_builder{ _ss }
    {}

    template <typename... Columns,
              typename =
                  std::enable_if_t<(common::traits::is_string_literal<Columns>() && ...)>>
    /**
     * @brief Append a parenthesized, comma-separated list of column names followed by " VALUES " to the internal stream.
     *
     * Accepts a variadic list of column identifiers (must be string-literal types). Produces output of the form:
     * "( col1, col2, ... ) VALUES " and returns the nested value-builder for chaining.
     *
     * @note This overload is constrained so all `Columns` are string-literal types; passing non-literal types is a compile-time error.
     *
     * @return query_value_builder& Reference to the internal query_value_builder for subsequent value construction.
     */
    query_value_builder& set_columns(Columns&... columns)
    {
        auto i = sizeof...(columns);
        _ss << "( ";
        ((_ss << columns << (i-- > 1 ? ", " : " ")), ...) << ") VALUES ";
        return _query_value_builder;
    }

private:
    std::stringstream&  _ss;
    query_value_builder _query_value_builder;
};

}  // namespace query_builders
}  // namespace queries
}  // namespace data_storage
}  // namespace rocpd
}  // namespace rocprofsys
