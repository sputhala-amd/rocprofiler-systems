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

#include "insert_query_builders.hpp"

namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{
namespace queries
{

struct table_insert_query
{
    /**
     * @brief Default-constructs a table_insert_query.
     *
     * Initializes the internal stringstream and constructs the query_columns_builder
     * to operate on that stream so subsequent calls (e.g., set_table_name) append
     * to the same underlying SQL fragment.
     */
    table_insert_query()
    : _query_columns_builder{ _ss }
    {}

    /**
     * @brief Start an INSERT query by setting the target table.
     *
     * Clears the internal query buffer and writes the leading fragment
     * "INSERT INTO <tableName> " into it.
     *
     * @param tableName Name of the table to insert into (may include schema). This function does not validate or escape the name.
     * @return query_builders::query_columns_builder& Reference to the columns/value builder to continue query construction.
     */
    query_builders::query_columns_builder& set_table_name(const std::string& tableName)
    {
        _ss.str("");
        _ss << "INSERT INTO " << tableName << " ";
        return _query_columns_builder;
    }

private:
    std::stringstream                     _ss;
    query_builders::query_columns_builder _query_columns_builder;
};

}  // namespace queries
}  // namespace data_storage
}  // namespace rocpd
}  // namespace rocprofsys
