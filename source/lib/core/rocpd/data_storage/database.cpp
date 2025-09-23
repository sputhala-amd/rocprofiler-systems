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

#include "database.hpp"
#include "common/md5sum.hpp"
#include "debug.hpp"
#include "node_info.hpp"

#include <config.hpp>
#include <fstream>
#include <regex>
#include <timemory/environment/types.hpp>
#include <timemory/utility/filepath.hpp>
#include <unistd.h>

namespace
{
/**
 * @brief Ensure the directory for a database file exists.
 *
 * Given a filesystem path to a database file, this function determines the
 * containing directory and creates it if it does not already exist.
 *
 * @param db_file Path to the database file whose directory should be created.
 */
void
create_directory_for_database_file(const std::string& db_file)
{
    auto _db_dirname = tim::filepath::dirname(db_file);
    if(!tim::filepath::direxists(_db_dirname))
    {
        tim::filepath::makedir(_db_dirname);
    }
}
}  // namespace
namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{
/**
 * @brief Returns the singleton instance of the database manager.
 *
 * Provides access to the single shared database object used by rocpd data storage.
 * The instance is created on first use and returned by reference.
 *
 * @return database& Reference to the singleton database instance.
 */
database&
database::get_instance()
{
    static database _instance;
    return _instance;
}

/**
 * @brief Construct and initialize the database singleton.
 *
 * Ensures the directory for the persistent database file ("rocpd.db") exists,
 * opens an in-memory temporary SQLite database (_sqlite3_db_temp) and the
 * persistent SQLite database at the resolved absolute path (_sqlite3_db),
 * and logs the chosen database path.
 *
 * The constructor performs filesystem preparation (creating the directory for
 * the DB file if needed) and opens both SQLite connections. If opening either
 * database fails, an error is propagated via validate_sqlite3_result.
 *
 * @throws std::runtime_error if opening the in-memory or persistent SQLite
 *         database fails.
 */
database::database()
{
    auto db_name     = std::string_view{ "rocpd.db" };
    auto abs_db_path = rocprofsys::get_database_absolute_path(db_name);
    create_directory_for_database_file(abs_db_path);
    ROCPROFSYS_VERBOSE(0, "Database: %s\r\n", abs_db_path.c_str());

    validate_sqlite3_result(sqlite3_open(":memory:", &_sqlite3_db_temp), "",
                            "database open failed!");
    validate_sqlite3_result(sqlite3_open(abs_db_path.c_str(), &_sqlite3_db), "",
                            "database open failed!");
}

/**
 * @brief Destructor that closes SQLite database connections.
 *
 * Closes the in-memory temporary database and the persistent database connections
 * held by this instance. After this returns, both _sqlite3_db_temp and
 * _sqlite3_db are no longer valid.
 */
database::~database()
{
    sqlite3_close(_sqlite3_db_temp);
    sqlite3_close(_sqlite3_db);
}

/**
 * @brief Load and apply SQL schema files into the in-memory database.
 *
 * This initializes the temporary (in-memory) SQLite database by locating,
 * preprocessing, and executing a set of schema SQL files:
 *  - rocpd_tables.sql
 *  - rocpd_views.sql
 *  - data_views.sql
 *  - marker_views.sql
 *  - summary_views.sql
 *
 * For each file the function:
 *  - Resolves the file path by first checking the environment variables
 *    `rocprofiler_systems_ROOT` or `ROCPROFSYS_ROOT` and, if present and valid,
 *    using a path under `${ROOT}/share/rocprofiler-systems/`. If no valid root
 *    is found, a built-in repository-relative path is used.
 *  - Reads the file contents and performs two placeholder substitutions:
 *    - `{{uuid}}` → `_<upid>` where `<upid>` is returned by get_upid().
 *    - `{{view_upid}}` → `` (empty string).
 *  - Executes the resulting SQL against the in-memory SQLite database handle
 *    (_sqlite3_db_temp).
 *
 * @throws std::runtime_error If any schema file cannot be opened.
 */
void
database::initialize_schema()
{
    auto get_file_path = [](const std::string_view filename) {
        auto _rocprofsys_root = tim::get_env<std::string>(
            "rocprofiler_systems_ROOT", tim::get_env<std::string>("ROCPROFSYS_ROOT", ""));
        if(!_rocprofsys_root.empty() &&
           tim::filepath::direxists(std::string(_rocprofsys_root)))
        {
            auto new_file_path = std::string(_rocprofsys_root)
                                     .append("/share/rocprofiler-systems/")
                                     .append(filename);
            if(tim::filepath::exists(new_file_path))
            {
                return new_file_path;
            }
        }
        return std::string(
                   "rocprofiler-systems/source/lib/core/rocpd/data_storage/schema/")
            .append(filename);
    };

    std::vector<std::string_view> schema_files = { "rocpd_tables.sql", "rocpd_views.sql",
                                                   "data_views.sql", "marker_views.sql",
                                                   "summary_views.sql" };

    // Process each schema file
    for(const auto& schema_file : schema_files)
    {
        auto          file_path = get_file_path(schema_file);
        std::ifstream file(file_path);
        if(!file.is_open())
        {
            throw std::runtime_error(
                std::string("Failed to open schema file ").append(file_path));
        }

        std::stringstream ss_query;
        ss_query << file.rdbuf();
        std::string query = ss_query.str();

        std::regex upid_pattern("\\{\\{uuid\\}\\}");
        std::regex view_upid_pattern("\\{\\{view_upid\\}\\}");

        query = std::regex_replace(query, upid_pattern, "_" + get_upid());
        query = std::regex_replace(query, view_upid_pattern, "");

        validate_sqlite3_result(
            sqlite3_exec(_sqlite3_db_temp, query.c_str(), 0, 0, 0), query.c_str(),
            std::string("Invalid schema file, init database failed!").append(file_path));
        file.close();
    }
}

/**
 * @brief Execute an SQL statement on the in-memory temporary database.
 *
 * Executes the provided SQL query against the internal in-memory SQLite database.
 * On failure the underlying SQLite error is reported via validate_sqlite3_result.
 *
 * @param query SQL statement to execute; may modify the temporary database.
 */
void
database::execute_query(const std::string& query)
{
    validate_sqlite3_result(sqlite3_exec(_sqlite3_db_temp, query.c_str(), 0, 0, 0),
                            "Failed to execute query - ", query);
}

/**
 * @brief Returns a stable unique process identifier (UPID).
 *
 * Computes an MD5-based identifier from the node identifier, current PID, and parent PID,
 * caches it on first call, and returns the cached hexadecimal string on subsequent calls.
 *
 * The value is stable for the lifetime of the process (cached in a static variable).
 *
 * @return std::string Hexadecimal UPID.
 */
std::string
database::get_upid()
{
    static std::string _upid = []() {
        auto n_info = node_info::get_instance();
        auto guid   = common::md5sum{ n_info.id, getpid(), getppid() };
        return guid.hexdigest();
    }();
    return _upid;
}

/**
 * @brief Returns the last rowid inserted into the in-memory temporary database.
 *
 * This returns the value of SQLite's last insert rowid for the temporary
 * in-memory connection used by this class. The value corresponds to the
 * most recent successful INSERT on the temporary database connection and is
 * 0 if no row has been inserted.
 *
 * @return size_t Last inserted row ID from the in-memory database.
 */
size_t
database::get_last_insert_id() const
{
    return sqlite3_last_insert_rowid(_sqlite3_db_temp);
}

/**
 * @brief Persist the in-memory temporary database to the on-disk database.
 *
 * Performs a full copy of the temporary in-memory SQLite database into the
 * persistent database using SQLite's backup API. If the backup cannot be
 * initialized (e.g., the backup handle is null), the function returns without
 * modifying the persistent database.
 *
 * @note This function has side effects on the persistent database file.
 */
void
database::flush()
{
    auto* backup = sqlite3_backup_init(_sqlite3_db, "main", _sqlite3_db_temp, "main");
    if(backup)
    {
        sqlite3_backup_step(backup, -1);  // Copy all pages
        sqlite3_backup_finish(backup);
    }
}

}  // namespace data_storage
}  // namespace rocpd
}  // namespace rocprofsys
