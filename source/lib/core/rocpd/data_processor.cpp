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

#include "data_processor.hpp"
#include "core/rocpd/data_storage/database.hpp"
#include "core/rocpd/data_storage/table_insert_query.hpp"
#include "debug.hpp"

namespace rocprofsys
{
namespace rocpd
{
/**
 * @brief Construct a data_processor and prepare the persistence layer.
 *
 * Initializes the underlying database schema, retrieves the current UPID,
 * and prepares all per-UPID prepared-statement executors and metadata used
 * for inserting ROCPD entities (events, PMC events, samples, regions,
 * kernel dispatches, memory copies, code objects, kernel symbols, args,
 * and memory allocation variants).
 *
 * Side effects:
 * - Calls the global database instance to initialize schema and obtain the UPID.
 * - Calls all initialize_*_stmt() helpers and initialize_metadata() to prepare
 *   statement executors and table-specific metadata for subsequent insertions.
 */
data_processor::data_processor()
{
    data_storage::database::get_instance().initialize_schema();
    _upid = data_storage::database::get_instance().get_upid();

    // Initialize event statement
    initialize_event_stmt();
    initialize_pmc_event_stmt();
    initialize_sample_stmt();
    initialize_region_stmt();
    initialize_kernel_dispatch_stmt();
    initialize_memory_copy_stmt();
    initialize_code_object_stmt();
    initialize_kernel_symbol_stmt();
    initialize_metadata();
    initialize_args_stmt();
    initialize_memory_alloc_stmt();
}

/**
 * @brief Returns the singleton instance of data_processor.
 *
 * The instance is created on first call and guaranteed to be initialized
 * in a thread-safe manner (per C++11 static local initialization).
 *
 * @return data_processor& Reference to the global data_processor instance.
 */
data_processor&
data_processor::get_instance()
{
    static data_processor _instance;
    return _instance;
}

/**
 * @brief Insert the processor's UPID into the per-UPID metadata table.
 *
 * Inserts a single row into the table rocpd_metadata_<upid> with columns
 * "tag" = "upid" and "value" = the instance's _upid, persisting the UPID
 * into the database.
 *
 * This function has the side effect of executing an INSERT against the
 * underlying database via the shared data_storage::database instance.
 */
void
data_processor::initialize_metadata()
{
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_metadata_" + _upid)
            .set_columns("tag", "value")
            .set_values("upid", _upid)
            .get_query_string());
}

/**
 * @brief Intern and persist a string, returning its database id.
 *
 * Looks up `str` in an internal thread-safe cache and returns the cached id if present.
 * Otherwise inserts the string into the per-UPID `rocpd_string_<upid>` table, caches
 * the generated id, and returns it.
 *
 * @param str Null-terminated string to intern and store.
 * @return size_t Database id assigned to the string.
 */
size_t
data_processor::insert_string(const char* str)
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    auto                        it = _string_map.find(str);
    if(it != _string_map.end()) return _string_map.at(str);

    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_string_" + _upid)
            .set_columns("guid", "string")
            .set_values(_upid, str)
            .get_query_string());

    const auto string_id = data_storage::database::get_instance().get_last_insert_id();
    _string_map.emplace(str, string_id);
    return string_id;
}

/**
 * @brief Insert node metadata into the per-UPID rocpd_info_node table.
 *
 * Inserts a row into the rocpd_info_node_<upid> table containing identifying and
 * descriptive information for a node. This operation writes the provided
 * values directly to the database; it does not modify in-memory caches.
 *
 * @param node_id Numeric identifier used as the row id in the table.
 * @param hash Node-specific hash value (used to detect/configure node identity).
 * @param machine_id Stable machine identifier/guid for the node.
 * @param system_name OS or distribution name.
 * @param hostname Hostname of the node.
 * @param release OS release string.
 * @param version OS version string.
 * @param hardware_name Hardware/board name.
 * @param domain_name Network domain name.
 */
void
data_processor::insert_node_info(size_t node_id, size_t hash, const char* machine_id,
                                 const char* system_name, const char* hostname,
                                 const char* release, const char* version,
                                 const char* hardware_name, const char* domain_name)
{
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_node_" + _upid)
            .set_columns("id", "guid", "hash", "machine_id", "system_name", "hostname",
                         "release", "version", "hardware_name", "domain_name")
            .set_values(node_id, _upid, hash, machine_id, system_name, hostname, release,
                        version, hardware_name, domain_name)
            .get_query_string());
}

/**
 * @brief Insert process metadata into the per-UPID process info table.
 *
 * Inserts a row into the rocpd_info_process_<upid> table containing identifiers
 * and metadata for a process running on a node.
 *
 * @param nid Node identifier where the process ran.
 * @param ppid Parent process identifier.
 * @param pid Process identifier (also stored as the table row id).
 * @param init Process initialization timestamp.
 * @param fini Process termination timestamp.
 * @param start Process start timestamp (lifespan or observation start).
 * @param end Process end timestamp (lifespan or observation end).
 * @param command Null-terminated command string used to launch the process.
 * @param environment Null-terminated environment string for the process.
 * @param extdata Optional extended metadata (stored as provided).
 */
void
data_processor::insert_process_info(size_t nid, size_t ppid, size_t pid, size_t init,
                                    size_t fini, size_t start, size_t end,
                                    const char* command, const char* environment,
                                    const char* extdata)
{
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_process_" + _upid)
            .set_columns("id", "guid", "nid", "ppid", "pid", "init", "fini", "start",
                         "end", "command", "environment", "extdata")
            .set_values(pid, _upid, nid, ppid, pid, init, fini, start, end, command,
                        environment, extdata)
            .get_query_string());
}

/**
 * @brief Insert an agent record into the per-UPID agent table and return its row id.
 *
 * Inserts a new row into rocpd_info_agent_<upid> with the provided agent metadata.
 *
 * @param node_id Node identifier (nid) associated with the agent.
 * @param pid Process identifier (pid) associated with the agent.
 * @param agent_type Human-readable agent type string (e.g., "GPU", "CPU", "ROCprof").
 * @param absolute_index Absolute agent index assigned by the system.
 * @param logical_index Logical index (user-visible or runtime ordering) for the agent.
 * @param type_index Index of this agent within its agent type/category.
 * @param uuid Agent UUID.
 * @param name Agent name (may be null).
 * @param model_name Model name string (may be null).
 * @param vendor_name Vendor name string (may be null).
 * @param product_name Product name string (may be null).
 * @param user_name User-facing name or owner (may be null).
 * @param extdata Optional extension/metadata blob (may be null).
 * @return size_t Database row id (last insert id) for the newly inserted agent.
 */
size_t
data_processor::insert_agent(size_t node_id, size_t pid, const char* agent_type,
                             size_t absolute_index, size_t logical_index,
                             size_t type_index, uint64_t uuid, const char* name,
                             const char* model_name, const char* vendor_name,
                             const char* product_name, const char* user_name,
                             const char* extdata)
{
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_agent_" + _upid)
            .set_columns("guid", "nid", "pid", "type", "absolute_index", "logical_index",
                         "type_index", "uuid", "name", "model_name", "vendor_name",
                         "product_name", "user_name", "extdata")
            .set_values(_upid, node_id, pid, agent_type, absolute_index, logical_index,
                        type_index, uuid, name, model_name, vendor_name, product_name,
                        user_name, extdata)
            .get_query_string());

    return data_storage::database::get_instance().get_last_insert_id();
}

/**
 * @brief Insert a new track row for the current UPID and cache its IDs.
 *
 * Inserts a new entry into the per-UPID rocpd_track table and records a mapping
 * from the track name to its persisted track ID and interned name ID. If a
 * track with the same name already exists in the in-memory cache, the function
 * logs a warning and returns without modifying the database.
 *
 * @param track_name Null-terminated track name to insert.
 * @param node_id Node identifier (nid) associated with the track.
 * @param process_id Process identifier (pid) associated with the track.
 * @param thread_id Optional thread identifier (tid); omitted when not applicable.
 * @param extdata Optional extended data string to store with the track.
 */
void
data_processor::insert_track(const char* track_name, size_t node_id, size_t process_id,
                             std::optional<size_t> thread_id, const char* extdata)
{
    if(_tracks.find(track_name) != _tracks.end())
    {
        ROCPROFSYS_WARNING(2, "Fail to add track %s, already exist!\n", track_name);
        return;
    }

    auto name_id = insert_string(track_name);

    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_track_" + _upid)
            .set_columns("guid", "nid", "pid", "tid", "name_id", "extdata")
            .set_values(_upid, node_id, process_id, thread_id, name_id, extdata)
            .get_query_string());

    auto track_id       = data_storage::database::get_instance().get_last_insert_id();
    _tracks[track_name] = track_name_map{ track_id, name_id };
}

/**
 * @brief Insert a performance-monitoring-counter (PMC) description and cache its id.
 *
 * Inserts a PMC descriptor row into the per-UPID table `rocpd_info_pmc_<upid>` and
 * records the resulting database id in the internal PMC descriptor map keyed by
 * (agent_id, name).
 *
 * If a descriptor with the same (agent_id, name) already exists in the map, the
 * function logs a warning and returns without modifying the database.
 *
 * @param node_id Node identifier associated with the PMC.
 * @param process_id Process identifier associated with the PMC.
 * @param agent_id Agent identifier that owns the PMC.
 * @param target_arch Target architecture string for the PMC.
 * @param event_code Numeric event code for the PMC.
 * @param instance_id Instance identifier for the PMC event.
 * @param name Human-readable name of the PMC (used as the descriptor key).
 * @param symbol Optional symbol name associated with the PMC.
 * @param description Short description of the PMC.
 * @param long_description Longer/free-form description of the PMC.
 * @param component Component string associated with the PMC.
 * @param units Units for the PMC value.
 * @param value_type Type/format of the PMC value.
 * @param block Optional block identifier/string for the PMC.
 * @param expression Optional expression describing derived PMCs.
 * @param is_constant Nonzero if the PMC is constant; zero otherwise.
 * @param is_derived Nonzero if the PMC is derived; zero otherwise.
 * @param extdata Optional extra data string stored with the descriptor.
 */
void
data_processor::insert_pmc_description(
    size_t node_id, size_t process_id, size_t agent_id, const char* target_arch,
    size_t event_code, size_t instance_id, const char* name, const char* symbol,
    const char* description, const char* long_description, const char* component,
    const char* units, const char* value_type, const char* block, const char* expression,
    uint32_t is_constant, uint32_t is_derived, const char* extdata)
{
    auto it = _pmc_descriptor_map.find({ agent_id, name });
    if(it != _pmc_descriptor_map.end())
    {
        ROCPROFSYS_WARNING(0,
                           "Insert PMC description failed! Error: PMC descriptor "
                           "(name:%s) (ID:%lu) already exist!\n",
                           name, agent_id);
        return;
    }
    data_storage::queries::table_insert_query query_builder;

    auto query =
        query_builder.set_table_name("rocpd_info_pmc_" + _upid)
            .set_columns("guid", "nid", "pid", "agent_id", "target_arch", "event_code",
                         "instance_id", "name", "symbol", "description",
                         "long_description", "component", "units", "value_type", "block",
                         "expression", "is_constant", "is_derived", "extdata")
            .set_values(_upid, node_id, process_id, agent_id, target_arch, event_code,
                        instance_id, name, symbol, description, long_description,
                        component, units, value_type, block, expression, is_constant,
                        is_derived, extdata)
            .get_query_string();
    data_storage::database::get_instance().execute_query(query);

    auto pmc_id = data_storage::database::get_instance().get_last_insert_id();
    _pmc_descriptor_map.emplace(
        std::pair<pmc_identifier, size_t>{ { agent_id, name }, pmc_id });
}

/**
 * @brief Record a PMC (performance-monitoring counter) sample for an event.
 *
 * Looks up the PMC description for the given agent and PMC name; if found, emits
 * a PMC event row into the database. If no matching PMC description exists, the
 * function logs a warning and returns without inserting.
 *
 * @param event_id Identifier of the associated event.
 * @param agent_id Identifier of the agent that produced the PMC.
 * @param pmc_name Null-terminated name of the PMC descriptor to look up.
 * @param value Numeric value sampled from the PMC.
 * @param extdata Optional extra metadata (nullable).
 */
void
data_processor::insert_pmc_event(size_t event_id, size_t agent_id, const char* pmc_name,
                                 double value, const char* extdata)
{
    ROCPROFSYS_VERBOSE(2,
                       "Insert PMC event: id %ld, agent id: %ld, pmc name: %s, value: "
                       "%lf, extdata: %s\n",
                       event_id, agent_id, pmc_name, value, extdata);
    auto it = _pmc_descriptor_map.find({ agent_id, pmc_name });
    if(it == _pmc_descriptor_map.end())
    {
        ROCPROFSYS_WARNING(0,
                           "Insert PMC event failed! Error: non-existing PMC description "
                           "agent id: %ld, pmc name: %s !\n",
                           agent_id, pmc_name);
        return;
    }

    const auto pmc_description_id = it->second;
    _insert_pmc_event_statement(_upid.c_str(), event_id, pmc_description_id, value,
                                extdata);
}

/**
 * @brief Insert a sample (timestamped event) for a named track into the database.
 *
 * Looks up the track by name; if the track is registered, appends a sample row
 * (track id, timestamp, event id, extdata) into the underlying per-UPID sample
 * table via the prepared statement executor. If the track name is not found the
 * function logs a warning and returns without inserting.
 *
 * @param track Null-terminated name of the track to which the sample belongs.
 * @param timestamp Sample timestamp (typically in ticks or nanoseconds).
 * @param event_id Identifier of the event associated with this sample.
 * @param extdata Optional JSON or auxiliary data (nullable C-string) stored with the sample.
 */
void
data_processor::insert_sample(const char* track, uint64_t timestamp, size_t event_id,
                              const char* extdata)
{
    ROCPROFSYS_VERBOSE(
        3, "Insert sample: track: %s, timestamp: %lu, event id: %ld, extdata: %s\n",
        track, timestamp, event_id, extdata);
    auto it = _tracks.find(track);
    if(it == _tracks.end())
    {
        ROCPROFSYS_WARNING(0, "Insert sample failed! Error: Unexisting track %s!\n",
                           track);
        return;
    }
    auto track_info = it->second;
    _insert_sample_statement(_upid.c_str(), track_info.track_id, timestamp, event_id,
                             extdata);
}

/**
 * @brief Insert an event row into the per‑UPID event table and return its database id.
 *
 * Inserts an event using the mapped category string id for the given category_id and the
 * provided stack/correlation information. Thread-safe; callers may be concurrent.
 *
 * @param category_id Numeric category identifier previously registered via insert_category().
 *                    Must exist in the processor's category map; otherwise a runtime_error is thrown.
 * @param stack_id Identifier of the call stack associated with the event.
 * @param parent_stack_id Identifier of the parent call stack (or 0 if none).
 * @param correlation_id Correlation id used to relate this event to others.
 * @param call_stack Serialized call‑stack data (string) to store with the event.
 * @param line_info Optional line/file information associated with the event.
 * @param extdata Optional extensible metadata stored with the event.
 *
 * @return size_t The last-inserted database row id for the new event.
 *
 * @throws std::runtime_error If category_id is not found in the processor's category map.
 */
size_t
data_processor::insert_event(size_t category_id, size_t stack_id, size_t parent_stack_id,
                             size_t correlation_id, const char* call_stack,
                             const char* line_info, const char* extdata)
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    auto                        it = _category_map.find(category_id);
    if(it == _category_map.end())
    {
        std::ostringstream oss;
        oss << "Insert event failed! Error: Unknown category id: " << category_id
            << " for UPID: " << _upid;
        throw std::runtime_error(oss.str());
    }

    ROCPROFSYS_VERBOSE(3, "Insert event category id: %ld, string id: %ld\n", category_id,
                       it->second);

    _insert_event_statement(_upid.c_str(), it->second, stack_id, parent_stack_id,
                            correlation_id, call_stack, line_info, extdata);
    return data_storage::database::get_instance().get_last_insert_id();
}

/**
 * @brief Prepare the SQL insert statement for event records.
 *
 * Initializes the prepared statement executor _insert_event_statement for the per-UPID
 * table "rocpd_event_<upid>" used to insert event rows.
 *
 * The prepared statement expects columns (in this parameter order):
 * 1. guid (const char*)
 * 2. category_id (size_t)
 * 3. stack_id (size_t)
 * 4. parent_stack_id (size_t)
 * 5. correlation_id (size_t)
 * 6. call_stack (const char*)
 * 7. line_info (const char*)
 * 8. extdata (const char*)
 */
void
data_processor::initialize_event_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_event_" + _upid)
                     .set_columns("guid", "category_id", "stack_id", "parent_stack_id",
                                  "correlation_id", "call_stack", "line_info", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_event_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, size_t, size_t,
                                       const char*, const char*, const char*>(query);
}

/**
 * @brief Prepare the SQL statement executor for inserting PMC events.
 *
 * Initializes a prepared INSERT statement for the per-UPID table
 * `rocpd_pmc_event_<upid>` with columns (guid, event_id, pmc_id, value, extdata)
 * and stores the resulting statement executor in the member
 * `_insert_pmc_event_statement`.
 *
 * The executor expects parameters in this order: guid (const char*),
 * event_id (size_t), pmc_id (size_t), value (double), extdata (const char*).
 */
void
data_processor::initialize_pmc_event_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_pmc_event_" + _upid)
                     .set_columns("guid", "event_id", "pmc_id", "value", "extdata")
                     .set_values('?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_pmc_event_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, double, const char*>(
                query);
}

/**
 * @brief Prepare the parameterized INSERT statement for samples.
 *
 * Initializes _insert_sample_statement with a prepared executor for inserting
 * rows into the per- UPID table `rocpd_sample_<upid>` with columns
 * (guid, track_id, timestamp, event_id, extdata). The statement uses five
 * positional placeholders and the executor expects parameters in the order:
 * `const char* guid, size_t track_id, uint64_t timestamp, size_t event_id,
 * const char* extdata`.
 */
void
data_processor::initialize_sample_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_sample_" + _upid)
                     .set_columns("guid", "track_id", "timestamp", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_sample_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, uint64_t, size_t,
                                       const char*>(query);
}

/**
 * @brief Prepare the SQL prepared-statement used to insert region records.
 *
 * Initializes _insert_region_statement with a prepared INSERT executor for the
 * per-UPC table "rocpd_region_<upid>" and the columns:
 * (guid, nid, pid, tid, start, end, name_id, event_id, extdata).
 *
 * The resulting statement accepts parameters in that exact column order and is
 * stored in the _insert_region_statement member for subsequent region insertions.
 */
void
data_processor::initialize_region_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_region_" + _upid)
                     .set_columns("guid", "nid", "pid", "tid", "start", "end", "name_id",
                                  "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_region_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, size_t, uint64_t,
                                       uint64_t, size_t, size_t, const char*>(query);
}

/**
 * @brief Prepare the SQL statement executor for inserting kernel dispatch records.
 *
 * Initializes a prepared statement executor stored in _insert_kernel_dispatch_statement that
 * inserts rows into the per-UPID table `rocpd_kernel_dispatch_<upid>`. The statement expects
 * values for the following columns (in order): guid, nid, pid, tid, agent_id, kernel_id,
 * dispatch_id, queue_id, stream_id, start, end, private_segment_size, group_segment_size,
 * workgroup_size_x, workgroup_size_y, workgroup_size_z, grid_size_x, grid_size_y, grid_size_z,
 * region_name_id, event_id, extdata.
 *
 * The created executor type matches the parameter sequence used when performing inserts:
 * const char*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
 * uint64_t, uint64_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
 * size_t, const char*.
 */
void
data_processor::initialize_kernel_dispatch_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_kernel_dispatch_" + _upid)
                     .set_columns("guid", "nid", "pid", "tid", "agent_id", "kernel_id",
                                  "dispatch_id", "queue_id", "stream_id", "start", "end",
                                  "private_segment_size", "group_segment_size",
                                  "workgroup_size_x", "workgroup_size_y",
                                  "workgroup_size_z", "grid_size_x", "grid_size_y",
                                  "grid_size_z", "region_name_id", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                                 '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_kernel_dispatch_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, size_t, size_t,
                                       size_t, size_t, size_t, size_t, uint64_t, uint64_t,
                                       size_t, size_t, size_t, size_t, size_t, size_t,
                                       size_t, size_t, size_t, size_t, const char*>(
                query);
}

/**
 * @brief Prepare the SQL insert statement executor for memory-copy records.
 *
 * Initializes _insert_memory_copy_statement with a prepared INSERT for the
 * per-upid table `rocpd_memory_copy_<upid>`. The prepared statement expects
 * values for the columns:
 * guid, nid, pid, tid, start, end, name_id,
 * dst_agent_id, dst_address, src_agent_id, src_address, size,
 * queue_id, stream_id, region_name_id, event_id, extdata
 * (placeholders used for each column) and is ready for execution with the
 * corresponding parameter types.
 */
void
data_processor::initialize_memory_copy_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_memory_copy_" + _upid)
                     .set_columns("guid", "nid", "pid", "tid", "start", "end", "name_id",
                                  "dst_agent_id", "dst_address", "src_agent_id",
                                  "src_address", "size", "queue_id", "stream_id",
                                  "region_name_id", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                                 '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_memory_copy_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, size_t, uint64_t,
                                       uint64_t, size_t, size_t, size_t, size_t, size_t,
                                       size_t, size_t, size_t, size_t, size_t,
                                       const char*>(query);
}

/**
 * @brief Prepare the SQL prepared-statement executor for inserting kernel symbol records.
 *
 * Initializes the internal prepared statement used to insert rows into the per-UPID
 * table `rocpd_info_kernel_symbol_<upid>`. The prepared statement accepts values for:
 * id, guid, nid, pid, code_object_id, kernel_name, display_name, kernel_object,
 * kernarg_segment_size, kernarg_segment_alignment, group_segment_size,
 * private_segment_size, sgpr_count, arch_vgpr_count, accum_vgpr_count, and extdata.
 */
void
data_processor::initialize_kernel_symbol_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto                                      query =
        query_builder.set_table_name("rocpd_info_kernel_symbol_" + _upid)
            .set_columns("id", "guid", "nid", "pid", "code_object_id", "kernel_name",
                         "display_name", "kernel_object", "kernarg_segment_size",
                         "kernarg_segment_alignment", "group_segment_size",
                         "private_segment_size", "sgpr_count", "arch_vgpr_count",
                         "accum_vgpr_count", "extdata")
            .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                        '?', '?', '?')
            .get_query_string();
    _insert_kernel_symbol_statement =
        data_storage::database::get_instance()
            .create_statement_executor<size_t, const char*, size_t, size_t, uint64_t,
                                       const char*, const char*, uint64_t, uint32_t,
                                       uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                       uint32_t, const char*>(query);
}

/**
 * @brief Prepare the SQL insert statement for code object records (per-UPID).
 *
 * Creates and stores a prepared statement executor used to insert rows into
 * the rocpd_info_code_object_<upid> table and binds parameters in the order:
 * id, guid, nid, pid, agent_id, uri, load_base, load_size, load_delta,
 * storage_type, extdata.
 */
void
data_processor::initialize_code_object_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto                                      query =
        query_builder.set_table_name("rocpd_info_code_object_" + _upid)
            .set_columns("id", "guid", "nid", "pid", "agent_id", "uri", "load_base",
                         "load_size", "load_delta", "storage_type", "extdata")
            .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
            .get_query_string();
    _insert_code_object_statement =
        data_storage::database::get_instance()
            .create_statement_executor<size_t, const char*, size_t, size_t, size_t,
                                       const char*, uint64_t, uint64_t, uint64_t,
                                       const char*, const char*>(query);
}

/**
 * @brief Prepare the SQL insert statement for argument records.
 *
 * Initializes _insert_args_statement with a prepared INSERT SQL for the per-UPID
 * table `rocpd_arg_<upid>`, covering columns (guid, event_id, position, type,
 * name, value, extdata). The created statement executor expects parameters in
 * the following order and types: const char* (guid), size_t (event_id),
 * size_t (position), const char* (type), const char* (name), const char* (value),
 * const char* (extdata).
 */
void
data_processor::initialize_args_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_arg_" + _upid)
                     .set_columns("guid", "event_id", "position", "type", "name", "value",
                                  "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_args_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, const char*,
                                       const char*, const char*, const char*>(query);
}

/**
 * @brief Prepare SQL insert statements for memory allocation records.
 *
 * Initializes two prepared statement executors used to insert rows into the
 * per-UPID table `rocpd_memory_allocate_<upid>`:
 * - one variant that includes an `agent_id` column, and
 * - one variant without `agent_id`.
 *
 * The statements accept the GUID, node/process/thread identifiers, memory
 * allocation metadata (type, level, start, end, address, size), queue/stream
 * identifiers, associated event id, and an extdata blob. The prepared
 * executors are stored in the members `_insert_memory_alloc_statement` and
 * `_insert_memory_alloc_no_agent_statement`.
 */
void
data_processor::initialize_memory_alloc_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_memory_allocate_" + _upid)
                     .set_columns("guid", "nid", "pid", "tid", "agent_id", "type",
                                  "level", "start", "end", "address", "size", "queue_id",
                                  "stream_id", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                                 '?', '?', '?', '?')
                     .get_query_string();
    _insert_memory_alloc_statement =
        data_storage::database::get_instance()
            .create_statement_executor<
                const char*, size_t, size_t, size_t, size_t, const char*, const char*,
                uint64_t, uint64_t, size_t, size_t, size_t, size_t, size_t, const char*>(
                query);

    // Statement without agent_id
    query = query_builder.set_table_name("rocpd_memory_allocate_" + _upid)
                .set_columns("guid", "nid", "pid", "tid", "type", "level", "start", "end",
                             "address", "size", "queue_id", "stream_id", "event_id",
                             "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                            '?', '?')
                .get_query_string();
    _insert_memory_alloc_no_agent_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, size_t, const char*,
                                       const char*, uint64_t, uint64_t, size_t, size_t,
                                       size_t, size_t, size_t, const char*>(query);
}

/**
 * @brief Insert an argument record for an event.
 *
 * Inserts an argument (type, name, value) at the given position for the specified
 * event into the per-UPID args table. The operation is thread-safe.
 *
 * @param event_id Database event identifier.
 * @param position Zero-based argument position.
 * @param type Argument type (C-string).
 * @param name Argument name (C-string).
 * @param value Argument value (C-string).
 * @param extdata Optional extra metadata (C-string) stored in the extdata column.
 */
void
data_processor::insert_args(size_t event_id, size_t position, const char* type,
                            const char* name, const char* value, const char* extdata)
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    _insert_args_statement(_upid.c_str(), event_id, position, type, name, value, extdata);
}

/**
 * @brief Insert stream metadata into the per-UPID stream info table.
 *
 * Inserts a stream row into the rocpd_info_stream_<upid> table and records the
 * stream_id in the local cache. If the stream_id has already been seen, the
 * call is a no-op.
 *
 * @param stream_id External stream identifier (used as the table id).
 * @param node_id Node identifier associated with the stream.
 * @param process_id Process identifier associated with the stream.
 * @param name Human-readable stream name.
 * @param extdata Optional extended data stored alongside the stream row.
 */
void
data_processor::insert_stream_info(size_t stream_id, size_t node_id, size_t process_id,
                                   const char* name, const char* extdata)
{
    if(_stream_ids.count(stream_id) > 0)
    {
        // ROCPROFSYS_WARNING(
        //     1, "Insert stream info failed! Error: Stream ID %ld already exists!\n",
        //     stream_id);
        return;
    }
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_stream_" + _upid)
            .set_columns("id", "guid", "nid", "pid", "name", "extdata")
            .set_values(stream_id, _upid, node_id, process_id, name, extdata)
            .get_query_string());
    _stream_ids.insert(stream_id);
}

/**
 * @brief Insert queue metadata into the per-UPID queue info table.
 *
 * If the given queue_id is already known, the call is a no-op.
 * Otherwise this inserts a row into rocpd_info_queue_<upid> with columns
 * (id, guid, nid, pid, name, extdata) and records the queue_id in the
 * in-memory _queue_ids set.
 *
 * @param queue_id Numeric identifier for the queue.
 * @param node_id Node identifier associated with the queue.
 * @param process_id Process identifier associated with the queue.
 * @param name Null-terminated display name for the queue.
 * @param extdata Optional null-terminated auxiliary data stored with the row.
 */
void
data_processor::insert_queue_info(size_t queue_id, size_t node_id, size_t process_id,
                                  const char* name, const char* extdata)
{
    if(_queue_ids.count(queue_id) > 0)
    {
        // ROCPROFSYS_WARNING(
        //     1, "Insert queue info failed! Error: Queue ID %ld already exists!\n",
        //     queue_id);
        return;
    }
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_queue_" + _upid)
            .set_columns("id", "guid", "nid", "pid", "name", "extdata")
            .set_values(queue_id, _upid, node_id, process_id, name, extdata)
            .get_query_string());
    _queue_ids.insert(queue_id);
}

/**
 * @brief Insert a code object record into the per-UPID code object table and cache its id.
 *
 * Inserts a code object row into rocpd_info_code_object_<upid> using the provided identifiers
 * and metadata, then records the code object id in the internal cache to prevent duplicate inserts.
 * If the given id is already known, the function returns without performing an insert.
 *
 * @param id Unique code object identifier (persisted as `id` in the DB).
 * @param node_id Node identifier (nid) associated with the code object.
 * @param process_id Process identifier (pid) associated with the code object.
 * @param agent_id Agent identifier associated with the code object.
 * @param uri URI or path of the code object.
 * @param ld_base Load base address of the code object.
 * @param ld_size Load size of the code object.
 * @param ld_delta Load delta of the code object.
 * @param storage_type Storage type string (e.g., how the code object is stored).
 * @param extdata Optional auxiliary metadata to store in the extdata column.
 */
void
data_processor::insert_code_object(size_t id, size_t node_id, size_t process_id,
                                   size_t agent_id, const char* uri, uint64_t ld_base,
                                   uint64_t ld_size, uint64_t ld_delta,
                                   const char* storage_type, const char* extdata)
{
    if(_code_object_ids.count(id) > 0)
    {
        // ROCPROFSYS_WARNING(
        //     1,
        //     "Insert code object info failed! Error: Code object ID %ld already
        //     exists!\n", id);
        return;
    }
    ROCPROFSYS_VERBOSE(2, "Insert code object with ID: %ld\n", id);
    std::lock_guard<std::mutex> lock(_data_mutex);
    _insert_code_object_statement(id, _upid.c_str(), node_id, process_id, agent_id, uri,
                                  ld_base, ld_size, ld_delta, storage_type, extdata);

    _code_object_ids.insert(id);
}

/**
 * @brief Insert a kernel symbol record if not already present.
 *
 * Inserts a kernel symbol row into the per-UPID kernel symbol table and records
 * the symbol ID in the local cache to prevent duplicate insertions. If the
 * symbol ID is already present, the function returns without modifying the
 * database.
 *
 * @param id Unique identifier for the kernel symbol (database id).
 * @param node_id Node identifier associated with the symbol.
 * @param process_id Process identifier associated with the symbol.
 * @param code_obj_id Identifier of the code object that contains the kernel.
 * @param name The kernel's internal name (used as stored name).
 * @param display_name Human-readable kernel name for display purposes.
 * @param kernel_obj Flag/identifier indicating kernel object presence/type.
 * @param kernarg_segmnt_size Size (bytes) of the kernel's kernarg segment.
 * @param kernarg_segment_alignment Alignment (bytes) requirement for kernargs.
 * @param group_segment_size Group segment size (bytes) for the kernel.
 * @param private_segment_size Private segment size (bytes) for the kernel.
 * @param sgrp_count Work-group/grid subgroup count related to the kernel.
 * @param arch_vgrp_count Architecture virtual group count for the kernel.
 * @param accum_vgrp_count Accumulated virtual group count for the kernel.
 * @param extdata Optional extra metadata as a JSON/text blob stored with the row.
 */
void
data_processor::insert_kernel_symbol(
    size_t id, size_t node_id, size_t process_id, uint64_t code_obj_id, const char* name,
    const char* display_name, uint32_t kernel_obj, uint32_t kernarg_segmnt_size,
    uint32_t kernarg_segment_alignment, uint32_t group_segment_size,
    uint32_t private_segment_size, uint32_t sgrp_count, uint32_t arch_vgrp_count,
    uint32_t accum_vgrp_count, const char* extdata)
{
    if(_kernel_sym_ids.count(id) > 0)
    {
        // ROCPROFSYS_WARNING(
        //     1,
        //     "Insert kernel symbol failed! Error: Kernel symbol ID %ld already
        //     exists!\n", id);
        return;
    }

    ROCPROFSYS_VERBOSE(2, "Insert kernel symbol: %s with ID: %ld\n", name, id);
    std::lock_guard<std::mutex> lock(_data_mutex);
    _insert_kernel_symbol_statement(
        id, _upid.c_str(), node_id, process_id, code_obj_id, name, display_name,
        kernel_obj, kernarg_segmnt_size, kernarg_segment_alignment, group_segment_size,
        private_segment_size, sgrp_count, arch_vgrp_count, accum_vgrp_count, extdata);

    _kernel_sym_ids.insert(id);
}

/**
 * @brief Record a mapping from an external category identifier to an internal string id.
 *
 * If the given category_id is already present, this function is a no-op. Otherwise it interns
 * the provided category name (via insert_string) and stores the resulting name id in the
 * internal category map for later lookups.
 *
 * Thread-safety: the insertion into the internal map is protected by the processor's mutex.
 *
 * @param category_id External category identifier (key used by incoming events).
 * @param name Human-readable category name to intern and associate with category_id.
 */
void
data_processor::insert_category(size_t category_id, const char* name)
{
    auto it = _category_map.find(category_id);
    if(it != _category_map.end())
    {
        // ROCPROFSYS_WARNING(
        //     1, "Insert category failed! Error: Category %s already exist!\n", name);
        return;
    }
    auto                        name_id = insert_string(name);
    std::lock_guard<std::mutex> lock(_data_mutex);
    ROCPROFSYS_VERBOSE(2, "Insert category: name: %s, id: %ld, name id: %ld\n", name,
                       category_id, name_id);
    _category_map.emplace(category_id, name_id);
}

/**
 * @brief Insert a region record associated with an event into the per-UPID region table.
 *
 * Thread-safe: acquires the internal data mutex before performing the insertion.
 *
 * @param node_id Host/node identifier where the region was recorded.
 * @param process_id Process identifier that owns the region.
 * @param thread_id Thread identifier that owns the region.
 * @param start Region start timestamp (ticks).
 * @param end Region end timestamp (ticks).
 * @param name_id Interned string id for the region name (from insert_string()).
 * @param event_id Identifier of the event that references this region.
 * @param extdata Optional auxiliary/contextual data (nullable C string).
 */
void
data_processor::insert_region(size_t node_id, size_t process_id, size_t thread_id,
                              uint64_t start, uint64_t end, size_t name_id,
                              size_t event_id, const char* extdata)
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    ROCPROFSYS_VERBOSE(2, "Insert region for event id: %ld\n", event_id);

    _insert_region_statement(_upid.c_str(), node_id, process_id, thread_id, start, end,
                             name_id, event_id, extdata);
}

/**
 * @brief Insert a kernel dispatch record into the per-UPID kernel dispatch table.
 *
 * Thread-safe: acquires the internal data mutex before performing the insert.
 *
 * @param node_id Node identifier where the dispatch occurred.
 * @param process_id Process identifier.
 * @param thread_id Thread identifier (tid).
 * @param agent_id Agent identifier that issued the kernel dispatch.
 * @param kernel_id Identifier of the kernel.
 * @param dispatch_id Identifier for this dispatch instance.
 * @param queue_id Queue identifier associated with the dispatch.
 * @param stream_id Stream identifier associated with the dispatch.
 * @param start Dispatch start timestamp.
 * @param end Dispatch end timestamp.
 * @param private_segment_size Kernel private segment size.
 * @param group_segment_size Kernel group segment size.
 * @param workgroup_size_x Workgroup size X dimension.
 * @param workgroup_size_y Workgroup size Y dimension.
 * @param workgroup_size_z Workgroup size Z dimension.
 * @param grid_size_x Grid size X dimension.
 * @param grid_size_y Grid size Y dimension.
 * @param grid_size_z Grid size Z dimension.
 * @param region_name_id String-id of the region name associated with the dispatch (from insert_string).
 * @param event_id Associated event identifier (if any).
 * @param extdata Optional external data string stored with the record.
 */
void
data_processor::insert_kernel_dispatch(
    size_t node_id, size_t process_id, size_t thread_id, size_t agent_id,
    size_t kernel_id, size_t dispatch_id, size_t queue_id, size_t stream_id,
    uint64_t start, uint64_t end, size_t private_segment_size, size_t group_segment_size,
    size_t workgroup_size_x, size_t workgroup_size_y, size_t workgroup_size_z,
    size_t grid_size_x, size_t grid_size_y, size_t grid_size_z, size_t region_name_id,
    size_t event_id, const char* extdata)
{
    std::lock_guard<std::mutex> lock(_data_mutex);

    ROCPROFSYS_VERBOSE(2, "Insert kernel dispatch for event id: %ld\n", event_id);

    _insert_kernel_dispatch_statement(
        _upid.c_str(), node_id, process_id, thread_id, agent_id, kernel_id, dispatch_id,
        queue_id, stream_id, start, end, private_segment_size, group_segment_size,
        workgroup_size_x, workgroup_size_y, workgroup_size_z, grid_size_x, grid_size_y,
        grid_size_z, region_name_id, event_id, extdata);
}

/**
 * @brief Insert a memory copy record for a region of execution.
 *
 * Records a memory copy event tied to a node/process/thread and associates it
 * with the prepared per-UPID memory_copy table. The call is thread-safe.
 *
 * @param node_id Node identifier (nid) associated with the copy.
 * @param process_id Process identifier (pid) associated with the copy.
 * @param thread_id Thread identifier (tid) associated with the copy.
 * @param start Start timestamp for the copy interval.
 * @param end End timestamp for the copy interval.
 * @param name_id Interned string id for the operation or region name.
 * @param dst_agent_id Destination agent identifier (if applicable).
 * @param dst_addr Destination address for the copy.
 * @param src_agent_id Source agent identifier (if applicable).
 * @param src_addr Source address for the copy.
 * @param size Number of bytes copied.
 * @param queue_id Queue identifier associated with the operation.
 * @param stream_id Stream identifier associated with the operation.
 * @param region_name_id Interned string id for the containing region name (if any).
 * @param event_id Associated event id (if any).
 * @param extdata Optional extra opaque metadata (nullable).
 */
void
data_processor::insert_memory_copy(size_t node_id, size_t process_id, size_t thread_id,
                                   uint64_t start, uint64_t end, size_t name_id,
                                   size_t dst_agent_id, size_t dst_addr,
                                   size_t src_agent_id, size_t src_addr, size_t size,
                                   size_t queue_id, size_t stream_id,
                                   size_t region_name_id, size_t event_id,
                                   const char* extdata)
{
    std::lock_guard<std::mutex> lock(_data_mutex);

    _insert_memory_copy_statement(_upid.c_str(), node_id, process_id, thread_id, start,
                                  end, name_id, dst_agent_id, dst_addr, src_agent_id,
                                  src_addr, size, queue_id, stream_id, region_name_id,
                                  event_id, extdata);
}

/**
 * @brief Insert a memory allocation record into the per-UPID memory allocation table.
 *
 * Adds a row describing a memory allocation window (start/end), address and size,
 * classification (type/level), and optional associations (agent, queue, stream, event).
 * If `agent_id` has a value, the variant that stores the agent identifier is used;
 * otherwise the agent-less variant is inserted.
 *
 * @param node_id Node identifier for the allocation.
 * @param process_id Process identifier for the allocation.
 * @param thread_id Thread identifier for the allocation.
 * @param agent_id Optional agent identifier; when present the record includes the agent.
 * @param type Allocation type string (e.g., "malloc", "hipMalloc").
 * @param level Memory level string (e.g., "host", "device").
 * @param start Allocation start timestamp.
 * @param end Allocation end timestamp.
 * @param address Allocation address.
 * @param size Allocation size in bytes.
 * @param queue_id Associated queue identifier (if applicable).
 * @param stream_id Associated stream identifier (if applicable).
 * @param event_id Associated event identifier (or 0 if none).
 * @param extdata Optional auxiliary/metadata string stored alongside the record.
 */
void
data_processor::insert_memory_alloc(size_t node_id, size_t process_id, size_t thread_id,
                                    std::optional<size_t> agent_id, const char* type,
                                    const char* level, uint64_t start, uint64_t end,
                                    size_t address, size_t size, size_t queue_id,
                                    size_t stream_id, size_t event_id,
                                    const char* extdata)
{
    if(agent_id.has_value())
    {
        _insert_memory_alloc_statement(_upid.c_str(), node_id, process_id, thread_id,
                                       agent_id.value(), type, level, start, end, address,
                                       size, queue_id, stream_id, event_id, extdata);
    }
    else
    {
        _insert_memory_alloc_no_agent_statement(
            _upid.c_str(), node_id, process_id, thread_id, type, level, start, end,
            address, size, queue_id, stream_id, event_id, extdata);
    }
}

/**
 * @brief Insert thread metadata (or return existing) and return its internal index.
 *
 * If the provided external thread_id has already been recorded, this returns the cached
 * internal thread index. Otherwise the function inserts a new row into the per-UPID
 * rocpd_info_thread_<upid> table and caches the database-assigned id before returning it.
 *
 * @param node_id Node identifier where the thread ran.
 * @param parent_process_id Parent process id for the thread.
 * @param process_id Process id owning the thread.
 * @param thread_id External thread identifier (used as the lookup key).
 * @param name Human-readable thread name.
 * @param start Thread start timestamp.
 * @param end Thread end timestamp.
 * @param extdata Optional auxiliary data stored with the thread record.
 * @return size_t Internal thread index (database row id) for the recorded thread.
 */
size_t
data_processor::insert_thread_info(size_t node_id, size_t parent_process_id,
                                   size_t process_id, size_t thread_id, const char* name,
                                   uint64_t start, uint64_t end, const char* extdata)
{
    auto it = _thread_id_map.find(thread_id);

    if(it != _thread_id_map.end())
    {
        return _thread_id_map.at(thread_id);
    }

    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_thread_" + _upid)
            .set_columns("guid", "nid", "ppid", "pid", "tid", "name", "start", "end",
                         "extdata")
            .set_values(_upid.c_str(), node_id, parent_process_id, process_id, thread_id,
                        name, start, end, extdata)
            .get_query_string());

    auto thread_idx = data_storage::database::get_instance().get_last_insert_id();
    _thread_id_map.emplace(thread_id, thread_idx);
    return thread_idx;
}

/**
 * @brief Flush all pending data to the underlying database.
 *
 * Ensures any buffered or queued writes held by the shared database instance are persisted.
 */
void
data_processor::flush()
{
    // Flush all pending data to the database
    data_storage::database::get_instance().flush();
}

}  // namespace rocpd
}  // namespace rocprofsys
