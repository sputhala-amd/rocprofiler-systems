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
#include <memory>
#include <mutex>
#include <sqlite3.h>
#include <sstream>
#include <stdexcept>

namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{
static std::mutex _mutex;
class database
{
public:
    static database& get_instance();

    /**
 * @brief Deleted copy constructor to prevent copying of the singleton database.
 *
 * Instances of `database` cannot be copied; this enforces singleton semantics and
 * avoids accidental duplication of the underlying SQLite handles and resources.
 */
database(database&)            = delete;
    /**
 * @brief Deleted copy assignment operator to enforce singleton semantics.
 *
 * Prevents assigning one database instance to another. The database is a singleton;
 * obtain the shared instance via database::get_instance().
 */
database& operator=(database&) = delete;

    void flush();

    ~database();

private:
    database();

    template <typename... Args>
    /**
     * @brief Validate a SQLite operation result and throw on error.
     *
     * Checks the provided SQLite return code and, if it indicates an error,
     * composes a detailed diagnostic message and throws a std::runtime_error.
     *
     * Behavior:
     * - If `sqlite3_error_code` is SQLITE_OK or SQLITE_DONE, the function returns.
     * - If `sqlite3_error_code` is SQLITE_CONSTRAINT, the function attempts to
     *   collect foreign-key constraint diagnostics by executing the supplied
     *   `query` against the auxiliary database handle and running
     *   `PRAGMA foreign_key_check`, appending any FK violation details to the
     *   error message.
     * - For other error codes the function appends the SQLite error string and the
     *   extended error message from the auxiliary DB handle, then throws.
     *
     * @param sqlite3_error_code SQLite return code to validate.
     * @param query The SQL query text associated with the operation (included in diagnostics).
     * @param args Optional contextual values that will be appended to the diagnostic message.
     *
     * @throws std::runtime_error Always throws for any non-success/non-DONE result; the
     *         exception message contains the composed diagnostics and SQLite error details.
     */
    inline void validate_sqlite3_result(int sqlite3_error_code, const char* query,
                                        Args&&... args)
    {
        std::stringstream ss;
        ss << "\n===========================================================\n";
        ss << "Database Error\n";
        ((ss << args << " "), ...);
        ss << "\nQuery: " << query << "\n";
        switch(sqlite3_error_code)
        {
            case SQLITE_OK:
            case SQLITE_DONE: return;
            case SQLITE_CONSTRAINT:
            {
                sqlite3_stmt* stmt;

                ss << "Constraint violation(s): " << "\n";

                sqlite3_exec(_sqlite3_db_temp, "PRAGMA foreign_keys = OFF;", nullptr,
                             nullptr, nullptr);
                sqlite3_exec(_sqlite3_db_temp, query, nullptr, nullptr, nullptr);
                sqlite3_exec(_sqlite3_db_temp, "PRAGMA foreign_keys = ON;", nullptr,
                             nullptr, nullptr);
                sqlite3_prepare_v2(_sqlite3_db_temp, "PRAGMA foreign_key_check", -1,
                                   &stmt, nullptr);
                int rc = 0;
                while((rc = sqlite3_step(stmt)) == SQLITE_ROW)
                {
                    const char* table  = (const char*) sqlite3_column_text(stmt, 0);
                    int         rowid  = sqlite3_column_int(stmt, 1);
                    const char* parent = (const char*) sqlite3_column_text(stmt, 2);
                    int         fkid   = sqlite3_column_int(stmt, 3);

                    ss << "  - " << "FK Violation - Table: " << (table ? table : "NULL")
                       << ", RowID: " << rowid
                       << ", Parent: " << (parent ? parent : "NULL") << ", FKID: " << fkid
                       << "\n";
                }

                sqlite3_finalize(stmt);
            }
            break;
            default:
            {
            }
            break;
        }
        ss << " [Sqlite3 error: " << sqlite3_errstr(sqlite3_error_code);
        ss << " (Extended error message: " << sqlite3_errmsg(_sqlite3_db_temp) << ")]";
        throw std::runtime_error(ss.str());
    }

    template <typename T, std::enable_if_t<!(common::traits::is_string_literal<T>() ||
                                             std::is_floating_point_v<std::decay_t<T>> ||
                                             std::is_same_v<std::decay_t<T>, int64_t> ||
                                             std::is_same_v<std::decay_t<T>, uint64_t> ||
                                             std::is_same_v<std::decay_t<T>, int32_t> ||
                                             std::is_same_v<std::decay_t<T>, uint32_t>),
                                           int> = 0>
    /**
     * @brief Fallback binding handler for unsupported parameter types.
     *
     * This overload is selected when no other bind_value overload matches the argument
     * type. It does not attempt to bind and instead reports the condition by throwing.
     *
     * @param stmt The prepared SQLite statement (unused here).
     * @param position 1-based placeholder index where a value would be bound (unused).
     * @param _value The value attempted to bind (unused).
     * @param query The SQL query associated with the binding attempt (unused).
     *
     * @throws std::runtime_error Always thrown to indicate the type cannot be bound.
     */
    inline void bind_value([[maybe_unused]] sqlite3_stmt* stmt,
                           [[maybe_unused]] int position, [[maybe_unused]] T& _value,
                           [[maybe_unused]] const std::string& query)
    {
        throw std::runtime_error("Unsupported type for binding!");
    }

    template <typename T,
              std::enable_if_t<common::traits::is_string_literal<T>(), int> = 0>
    /**
     * @brief Bind a text value to a prepared SQLite statement.
     *
     * Binds the provided text value into the prepared statement parameter at the
     * given 1-based position using SQLITE_STATIC lifetime. On any SQLite error the
     * call is forwarded to validate_sqlite3_result which will throw with a
     * descriptive message.
     *
     * @param stmt Prepared SQLite statement to bind into.
     * @param position 1-based placeholder index within the statement.
     * @param _value Text value to bind (accepted as a forwarding reference).
     * @param query SQL query text used to augment error diagnostics.
     */
    inline void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                           const std::string& query)
    {
        validate_sqlite3_result(
            sqlite3_bind_text(stmt, position, _value, -1, SQLITE_STATIC), query.c_str(),
            "Failed to bind text! Position: ", position, ", Value: ", _value);
    }

    template <typename T,
              std::enable_if_t<std::is_floating_point_v<std::decay_t<T>>, int> = 0>
    /**
     * @brief Binds a floating-point value to a prepared SQLite statement parameter.
     *
     * Binds the provided floating-point value into the given sqlite3_stmt at the
     * specified 1-based parameter position. On failure this delegates to
     * validate_sqlite3_result which will throw a std::runtime_error with a message
     * containing the query and binding context.
     *
     * @param stmt Prepared SQLite statement to bind into.
     * @param position 1-based index of the parameter placeholder in the statement.
     * @param _value Floating-point value to bind (e.g., float, double).
     * @param query SQL query text used to enrich error messages if binding fails.
     *
     * @throws std::runtime_error If sqlite3_bind_double reports an error.
     */
    inline void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                           const std::string& query)
    {
        validate_sqlite3_result(
            sqlite3_bind_double(stmt, position, _value), query.c_str(),
            "Failed to bind double! Position: ", position, ", Value: ", _value);
    }

    template <typename T, std::enable_if_t<std::is_same_v<std::decay_t<T>, int64_t> ||
                                               std::is_same_v<std::decay_t<T>, uint64_t>,
                                           int> = 0>
    /**
     * @brief Bind a 64-bit integer value to a prepared SQLite statement.
     *
     * Binds an `int64_t` or `uint64_t` value into the given `sqlite3_stmt` at the specified
     * (1-based) placeholder index. The supplied `query` is used only for error reporting.
     *
     * @param stmt Prepared SQLite statement.
     * @param position 1-based placeholder index in the statement where the value will be bound.
     * @param _value 64-bit integer value to bind (signed or unsigned).
     * @param query SQL query text used to provide context if an error occurs.
     *
     * @throws std::runtime_error If the underlying SQLite bind operation fails; the exception
     *         message includes the provided `query` and binding context.
     */
    inline void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                           const std::string& query)
    {
        validate_sqlite3_result(sqlite3_bind_int64(stmt, position, _value), query.c_str(),
                                "Failed to bind int64_t/uint64_t! Position: ", position,
                                ", Value: ", _value);
    }

    template <typename T, std::enable_if_t<std::is_same_v<std::decay_t<T>, int32_t> ||
                                               std::is_same_v<std::decay_t<T>, uint32_t>,
                                           int> = 0>
    /**
     * @brief Bind a 32-bit integer value to a prepared SQLite statement parameter.
     *
     * Binds an `int32_t` or `uint32_t` value into the given prepared statement at the
     * specified parameter index and validates the SQLite result.
     *
     * @param stmt Prepared SQLite statement handle.
     * @param position 1-based parameter index in the prepared statement where the value will be bound.
     * @param _value Integer value to bind (int32_t or uint32_t).
     * @param query SQL query text used to annotate error messages if binding fails.
     *
     * @throws std::runtime_error If sqlite3_bind_int reports an error (validated via validate_sqlite3_result).
     */
    inline void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                           const std::string& query)
    {
        validate_sqlite3_result(sqlite3_bind_int(stmt, position, _value), query.c_str(),
                                "Failed to bind int32_t/uint32_t! Position: ", position,
                                ", Value: ", _value);
    }

public:
    void initialize_schema();

    void execute_query(const std::string& query);

    size_t get_last_insert_id() const;

    /**
     * This function prepares an SQLite statement based on the provided SQL query and
     * returns a lambda that can execute the prepared statement, binding the provided
     * values to the respective placeholders in the query.
     */
    template <typename... Values>
    /**
     * @brief Prepare a parameterized SQLite statement and return a reusable executor.
     *
     * Prepares `query` on the temporary SQLite handle and returns a callable that,
     * when invoked with a sequence of values matching the statement parameters,
     * will bind those values (1-based positions), execute the statement, and reset
     * it for reuse. Execution is serialized with a mutex to make invocations
     * thread-safe. Any SQLite error encountered during prepare, bind, or step is
     * forwarded via validate_sqlite3_result as a std::runtime_error.
     *
     * @tparam Values Types of the parameters to bind; the returned callable accepts `Values...`.
     * @param query SQL statement to prepare (may contain `?` parameter placeholders).
     * @return A callable with signature `void(Values...)` that binds the arguments,
     *         executes the statement, and resets it for subsequent use.
     */
    auto create_statement_executor(const std::string& query)
    {
        sqlite3_stmt* p_stmt;
        validate_sqlite3_result(
            sqlite3_prepare_v2(_sqlite3_db_temp, query.c_str(), -1, &p_stmt, nullptr),
            query.c_str(), "Failed to create statement!");
        std::shared_ptr<sqlite3_stmt> stmt{ p_stmt, sqlite3_finalize };

        return [stmt, query, this](Values... value) {
            std::lock_guard lock{ _mutex };
            int             position = 1;

            ((bind_value(stmt.get(), position++, value, query)), ...);

            validate_sqlite3_result(sqlite3_step(stmt.get()), query.c_str(),
                                    "Failed to execute step!\n", "Values: ", value...);
            sqlite3_reset(stmt.get());
        };
    }

    static std::string get_upid();

private:
    sqlite3* _sqlite3_db{ nullptr };
    sqlite3* _sqlite3_db_temp{ nullptr };
};

}  // namespace data_storage
}  // namespace rocpd
}  // namespace rocprofsys
