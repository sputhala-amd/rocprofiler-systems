// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/components/comm_data.hpp"
#include "core/agent_manager.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/node_info.hpp"
#include "core/perfetto.hpp"
#include "core/rocpd/data_processor.hpp"
#include "library/tracing.hpp"

#include <timemory/units.hpp>

namespace rocprofsys
{
namespace component
{
namespace
{
template <typename Tp, typename... Args>
/**
 * @brief Emit a Perfetto counter sample for the given Track type, accumulating values.
 *
 * If Perfetto tracing is enabled and the system state is Active, this function lazily
 * creates a counter track for the Track type `Tp` (labelled by `Tp::label`) and emits
 * a TRACE_COUNTER sample with a thread-safe running accumulation of the provided value.
 *
 * The counter track is created once (per process) with unit "bytes". The accumulated
 * value is updated under a mutex and the sample timestamp is taken via
 * rocprofsys::tracing::now<uint64_t>(). The emitted counter uses `Tp::value` as the
 * counter identifier.
 *
 * Requirements:
 * - `Tp` must provide a static `label` (string-like) and a static `value` (identifier).
 *
 * @param _val Amount to add to the running counter (interpreted in bytes); the emitted
 *             sample reports the post-addition accumulated value.
 */
void
write_perfetto_counter_track(uint64_t _val)
{
    using counter_track = rocprofsys::perfetto_counter_track<Tp>;

    if(rocprofsys::get_use_perfetto() &&
       rocprofsys::get_state() == rocprofsys::State::Active)
    {
        auto _emplace = [](const size_t _idx) {
            if(!counter_track::exists(_idx))
            {
                std::string _label = (_idx > 0)
                                         ? JOIN(" ", Tp::label, JOIN("", '[', _idx, ']'))
                                         : Tp::label;
                counter_track::emplace(_idx, _label, "bytes");
            }
        };

        const size_t          _idx = 0;
        static std::once_flag _once{};
        std::call_once(_once, _emplace, _idx);

        static std::mutex _mutex{};
        static uint64_t   value = 0;
        uint64_t          _now  = 0;
        {
            std::unique_lock<std::mutex> _lk{ _mutex };
            _now = rocprofsys::tracing::now<uint64_t>();
            _val = (value += _val);
        }

        TRACE_COUNTER(Tp::value, counter_track::at(_idx, 0), _now, _val);
    }
}
}  // namespace

namespace
{
/**
 * @brief Returns the singleton rocpd::data_processor instance.
 *
 * Provides access to the global data processor used for rocpd-based
 * communication metric registration and event processing.
 *
 * @return rocpd::data_processor& Reference to the process-global singleton instance.
 */
rocpd::data_processor&
get_data_processor()
{
    return rocpd::data_processor::get_instance();
}

/**
 * @brief Register communication-related categories with the rocpd data processor.
 *
 * Ensures the comm_data category is inserted into the global rocpd::data_processor exactly once.
 * Conditionally registers the MPI and ROCm/RCCL categories when the corresponding build
 * flags (ROCPROFSYS_USE_MPI, ROCPROFSYS_USE_RCCL) are defined.
 *
 * This function is idempotent: subsequent calls are no-ops after the first successful initialization.
 */
void
rocpd_initialize_comm_data_categories()
{
    static bool _is_initialized = false;
    if(_is_initialized) return;

    get_data_processor().insert_category(category_enum_id<category::comm_data>::value,
                                         trait::name<category::comm_data>::value);
#if defined(ROCPROFSYS_USE_MPI)
    get_data_processor().insert_category(category_enum_id<category::mpi>::value,
                                         trait::name<category::mpi>::value);
#endif
#if defined(ROCPROFSYS_USE_RCCL)
    get_data_processor().insert_category(category_enum_id<category::rocm_rccl>::value,
                                         trait::name<category::rocm_rccl>::value);
#endif
    _is_initialized = true;
}

template <typename Track>
/**
 * @brief Initialize a perf tracking track for the specified Track type (once).
 *
 * Inserts a track into the rocpd data processor using Track::label, the current
 * node id from node_info, and the current process id. Initialization is
 * performed exactly once (thread-safe).
 *
 * @tparam Track Track type that must expose a static `label` member (const char*).
 *
 * Side effects: registers a track in the global data_processor.
 */
void
rocpd_initialize_track()
{
    auto& n_info      = node_info::get_instance();
    auto  thread_id   = std::nullopt;
    auto  _init_track = [&](const char* label) {
        ROCPROFSYS_VERBOSE(3, "INSERT_TRACK label: %s, node ID: %d, Process ID: %d",
                            label, n_info.id, getpid());
        get_data_processor().insert_track(label, n_info.id, getpid(), thread_id);
    };

    static std::once_flag _once{};
    std::call_once(_once, _init_track, Track::label);
}

/**
 * @brief Register PMC (performance monitoring counter) descriptions for communication data.
 *
 * @details
 * Initializes and inserts PMC metadata entries into the internal rocpd data processor for
 * communication-related metrics (bytes sent/received). The function determines node and
 * agent context (node id, process id, agent base id) and registers PMC descriptions for
 * MPI and/or RCCL communication tracks when those integrations are enabled at compile time.
 *
 * The registered descriptions include labels, human-readable descriptions, category
 * descriptions, units ("bytes"/"ABS"), and other metadata consumed by the rocpd
 * data_processor for later event/sample insertion.
 *
 * This function has no parameters and no return value. It relies on node_info and
 * agent_manager singletons and is guarded by compile-time flags:
 * - When ROCPROFSYS_USE_MPI is defined, inserts entries for comm_data::mpi_send and comm_data::mpi_recv.
 * - When ROCPROFSYS_USE_RCCL is defined, inserts entries for rccl_send and rccl_recv.
 *
 * @note The implementation assumes a CPU agent with device id 0 when resolving the agent base id.
 */
void
rocpd_initialize_comm_data_pmc()
{
    [[maybe_unused]] auto& data_processor = get_data_processor();
    // find the proper values for a following definitions
    [[maybe_unused]] size_t                EVENT_CODE       = 0;
    [[maybe_unused]] size_t                INSTANCE_ID      = 0;
    [[maybe_unused]] constexpr const char* LONG_DESCRIPTION = "";
    [[maybe_unused]] constexpr const char* COMPONENT        = "";
    [[maybe_unused]] constexpr const char* BLOCK            = "";
    [[maybe_unused]] constexpr const char* EXPRESSION       = "";
    [[maybe_unused]] constexpr const char* MSG              = "bytes";
    [[maybe_unused]] constexpr const auto* TARGET_ARCH      = "CPU";
    auto                                   ni               = node_info::get_instance();
    constexpr const auto                   DEVICE_ID = 0;  // Assuming CPU device ID is 0

    auto&                 _agent_manager = agent_manager::get_instance();
    [[maybe_unused]] auto base_id =
        _agent_manager.get_agent_by_id(DEVICE_ID, agent_type::CPU).base_id;

#if defined(ROCPROFSYS_USE_MPI)
    data_processor.insert_pmc_description(
        ni.id, getpid(), base_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
        comm_data::mpi_send::label, "Tracks MPI Send communication data sizes",
        trait::name<category::mpi>::description, LONG_DESCRIPTION, COMPONENT, MSG, "ABS",
        BLOCK, EXPRESSION, 0, 0);

    data_processor.insert_pmc_description(
        ni.id, getpid(), base_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
        comm_data::mpi_recv::label, "Tracks MPI Receive communication data sizes",
        trait::name<category::mpi>::description, LONG_DESCRIPTION, COMPONENT, MSG, "ABS",
        BLOCK, EXPRESSION, 0, 0);
#endif
#if defined(ROCPROFSYS_USE_RCCL)
    data_processor.insert_pmc_description(
        ni.id, getpid(), base_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID, rccl_send::label,
        "Tracks RCCL Send communication data sizes",
        trait::name<category::rocm_rccl>::description, LONG_DESCRIPTION, COMPONENT, MSG,
        "ABS", BLOCK, EXPRESSION, 0, 0);

    data_processor.insert_pmc_description(
        ni.id, getpid(), base_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID, rccl_recv::label,
        "Tracks RCCL Receive communication data sizes",
        trait::name<category::rocm_rccl>::description, LONG_DESCRIPTION, COMPONENT, MSG,
        "ABS", BLOCK, EXPRESSION, 0, 0);
#endif
}

template <typename Track>
/**
 * @brief Record CPU-side communication byte counts as a rocpd PMC event and sample.
 *
 * Inserts a per-call PMC event and a time-stamped sample into the rocpd data processor
 * for the CPU agent associated with the given device. The function maintains a
 * process-wide, thread-safe accumulator of bytes (static) that is incremented by
 * the provided `bytes` on each call and the accumulated value is emitted.
 *
 * @tparam Track Type providing a `static const char* label` used as the PMC/event name.
 * @param device_id Device identifier used to look up the CPU agent whose device_id is attributed to the sample.
 * @param bytes Number of bytes to add to the running accumulator; the emitted value is the accumulator after adding this amount.
 */
void
rocpd_process_cpu_usage_events(const uint32_t device_id, int bytes)
{
    auto& data_processor = get_data_processor();
    auto  event_id       = data_processor.insert_event(
        category_enum_id<category::comm_data>::value, 0, 0, 0);

    auto& agents = agent_manager::get_instance();
    auto  agent  = agents.get_agent_by_id(device_id, agent_type::CPU);

    auto insert_event_and_sample = [&](const char* name, uint64_t timestamp,
                                       double value) {
        data_processor.insert_pmc_event(event_id, agent.device_id, name, value);
        data_processor.insert_sample(name, timestamp, event_id);
    };

    static std::mutex _mutex{};
    static uint64_t   value = 0;
    uint64_t          _now  = 0;
    {
        std::unique_lock<std::mutex> _lk{ _mutex };
        _now  = rocprofsys::tracing::now<uint64_t>();
        bytes = (value += bytes);
    }

    insert_event_and_sample(Track::label, _now, bytes);
}

}  /**
 * @brief Initialize rocpd communication-data integrations when enabled.
 *
 * If rocpd support is active (get_use_rocpd() returns true), registers
 * communication data categories and PMD/PMC descriptions with the
 * rocpd::data_processor by calling the internal initialization helpers.
 *
 * This function has no effect when rocpd is disabled.
 */

void
comm_data::start()
{
    if(get_use_rocpd())
    {
        rocpd_initialize_comm_data_categories();
        rocpd_initialize_comm_data_pmc();
    }
}

/**
 * @brief Perform early initialization for comm_data.
 *
 * Ensures the component's configuration is applied by invoking configure().
 */
void
comm_data::preinit()
{
    configure();
}

void
comm_data::global_finalize()
{
    configure();
}

void
comm_data::configure()
{
    static bool _once = false;
    if(_once) return;
    _once = true;

    comm_data_tracker_t::label()        = "comm_data";
    comm_data_tracker_t::description()  = "Tracks MPI/RCCL communication data sizes";
    comm_data_tracker_t::display_unit() = "MB";
    comm_data_tracker_t::unit()         = units::megabyte;

    auto _fmt_flags = comm_data_tracker_t::get_format_flags();
    _fmt_flags &= (std::ios_base::fixed & std::ios_base::scientific);
    _fmt_flags |= (std::ios_base::scientific);
    comm_data_tracker_t::set_precision(3);
    comm_data_tracker_t::set_format_flags(_fmt_flags);
}

#if defined(ROCPROFSYS_USE_MPI)
/**
 * @brief Audit an outgoing MPI_Send and record communicated byte counts.
 *
 * Records the total bytes sent (count * sizeof(datatype)) to enabled backends:
 * - Emits a Perfetto counter track for mpi_send.
 * - If rocpd is enabled, initializes the mpi_send rocpd track and posts a CPU-usage event sample.
 * - If Timemory is enabled, updates per-tool trackers labeled by destination and tag.
 *
 * @param _data Contains caller metadata (e.g., tool_id used as tracker name in Timemory).
 * @param count Number of elements sent.
 * @param datatype MPI datatype of each element; its size is used to compute total bytes. If size is zero, the function returns without action.
 * @param dst MPI destination rank (used in Timemory tracker labels).
 * @param tag MPI message tag (used in Timemory tracker labels).
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    write_perfetto_counter_track<mpi_send>(count * _size);

    if(get_use_rocpd())
    {
        rocpd_initialize_track<mpi_send>();
        rocpd_process_cpu_usage_events<mpi_send>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _a{ _name };
        add(_a, count * _size);
        tracker_t _b{ JOIN('/', _name, JOIN('=', "dst", dst)) };
        add(_b, count * _size);
        add(JOIN('/', _name, JOIN('=', "dst", dst), JOIN('=', "tag", tag)),
            count * _size);
    }
}

/**
 * @brief Handle an incoming MPI_Recv event: record communicated byte count to enabled backends.
 *
 * Records the total byte size (count * sizeof(datatype)) for an MPI receive and, depending
 * runtime flags, emits a Perfetto counter, registers/updates an rocpd track and per-event
 * CPU-usage sample, and adds timemory trackers with per-peer and per-tag labels.
 *
 * If the computed element size is zero the function returns immediately.
 *
 * @param _data Container with metadata for the intercepted call; its `tool_id` is used
 *              as the base name for timemory trackers.
 * @param count Number of elements received.
 * @param datatype MPI datatype of each element (used to compute element size).
 * @param dst    Peer rank associated with the receive (used in tracker labels).
 * @param tag    MPI tag associated with the message (used in tracker labels).
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Status*)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<mpi_recv>(count * _size);

    if(get_use_rocpd())
    {
        rocpd_initialize_track<mpi_recv>();
        rocpd_process_cpu_usage_events<mpi_recv>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _a{ _name };
        add(_a, count * _size);
        tracker_t _b{ JOIN('/', _name, JOIN('=', "dst", dst)) };
        add(_b, count * _size);
        add(JOIN('/', _name, JOIN('=', "dst", dst), JOIN('=', "tag", tag)),
            count * _size);
    }
}

/**
 * @brief Audit handler for non-blocking MPI send operations (MPI_Isend).
 *
 * Computes the total message size (count * sizeof(datatype)) and, if non-zero,
 * conditionally records the transfer via Perfetto counters, the rocpd data
 * processor, and timemory trackers according to the runtime configuration.
 *
 * @param _data Metadata about the intercepted call; its `tool_id` is used as the base name
 *              for timemory trackers.
 * @param count Number of elements in the send.
 * @param datatype MPI datatype of each element; its size is obtained via mpi_type_size().
 * @param dst Destination MPI rank for the send (used in timemory tracker labels).
 * @param tag MPI message tag (used in timemory tracker labels).
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Request*)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<mpi_send>(count * _size);

    if(get_use_rocpd())
    {
        rocpd_initialize_track<mpi_send>();
        rocpd_process_cpu_usage_events<mpi_send>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _a{ _name };
        add(_a, count * _size);
        tracker_t _b{ JOIN('/', _name, JOIN('=', "dst", dst)) };
        add(_b, count * _size);
        add(JOIN('/', _name, JOIN('=', "dst", dst), JOIN('=', "tag", tag)),
            count * _size);
    }
}

/**
 * @brief Handle an incoming MPI_Irecv audit event and record communicated byte counts.
 *
 * This records the total byte size (count * size_of(datatype)) for an incoming non-blocking
 * MPI receive using the enabled telemetry backends:
 * - If Perfetto is enabled, emits a counter update for the mpi_recv track.
 * - If the rocpd data path is enabled, ensures the mpi_recv track is initialized and
 *   records a per-event CPU usage sample.
 * - If Timemory is enabled, updates per-tool trackers using _data.tool_id and adds
 *   additional trackers annotated with destination and tag.
 *
 * If the MPI datatype size is zero the function returns immediately and performs no recording.
 *
 * @param _data Contains metadata for the audited call; _data.tool_id is used as the base
 *              tracker name for Timemory updates.
 * @param count Number of elements received.
 * @param datatype MPI datatype of each element; its size is used to compute total bytes.
 * @param dst    Destination/peer rank associated with the receive (used for labeling).
 * @param tag    MPI tag associated with the receive (used for labeling).
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Request*)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<mpi_recv>(count * _size);

    if(get_use_rocpd())
    {
        rocpd_initialize_track<mpi_recv>();
        rocpd_process_cpu_usage_events<mpi_recv>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _a{ _name };
        add(_a, count * _size);
        tracker_t _b{ JOIN('/', _name, JOIN('=', "dst", dst)) };
        add(_b, count * _size);
        add(JOIN('/', _name, JOIN('=', "dst", dst), JOIN('=', "tag", tag)),
            count * _size);
    }
}

/**
 * @brief Handle an incoming MPI_Bcast event for communication-data tracking.
 *
 * Records the total bytes communicated (count * sizeof(datatype)) for an MPI_Bcast.
 * If the datatype size is zero, the call is a no-op. Depending on runtime flags,
 * the function will:
 * - emit a Perfetto counter sample for mpi_send,
 * - initialize and emit a rocpd per-event CPU-usage sample for mpi_send,
 * - record Timemory trackers named by the originating tool id and a "root" variant.
 *
 * The Timemory trackers use _data.tool_id as the base name and append "root=<root>"
 * for the root-specific breakdown.
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int root, MPI_Comm)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<mpi_send>(count * _size);

    if(get_use_rocpd())
    {
        rocpd_initialize_track<mpi_send>();
        rocpd_process_cpu_usage_events<mpi_send>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count * _size);
        add(JOIN('/', _name, JOIN('=', "root", root)), count * _size);
    }
}

/**
 * @brief Handle an intercepted MPI_Allreduce (incoming) event.
 *
 * Computes the total transferred byte size (count * size_of(datatype)) and, if nonzero:
 * - Emits Perfetto counter updates for both send and receive tracks when Perfetto is enabled.
 * - Lazily initializes rocpd tracks and records per-event CPU-usage samples for both send and receive when rocpd is enabled.
 * - Records aggregated data with timemory when timemory is enabled.
 *
 * @param _data Gotcha wrapper metadata for the intercepted MPI call (used when recording timemory data).
 * @param count Number of elements in the Allreduce operation; combined with `datatype` to compute byte size.
 * @param datatype MPI datatype of each element; its size is used to compute the total byte payload.
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, void*, int count,
                 MPI_Datatype datatype, MPI_Op, MPI_Comm)
{
    int _size = mpi_type_size(datatype);
    if(_size == 0) return;

    if(get_use_perfetto())
    {
        write_perfetto_counter_track<mpi_recv>(count * _size);
        write_perfetto_counter_track<mpi_send>(count * _size);
    }

    if(get_use_rocpd())
    {
        rocpd_initialize_track<mpi_send>();
        rocpd_initialize_track<mpi_recv>();
        rocpd_process_cpu_usage_events<mpi_recv>(0, count * _size);
        rocpd_process_cpu_usage_events<mpi_send>(0, count * _size);
    }

    if(rocprofsys::get_use_timemory()) add(_data, count * _size);
}

/**
 * @brief Audit an MPI_Sendrecv operation and record communication metrics.
 *
 * Records send and receive byte counts for an MPI_Sendrecv call. If enabled,
 * emits Perfetto counters, enqueues per-event CPU-usage samples via the rocpd
 * data processor, and updates timemory trackers with aggregated and
 * per-direction breakdowns (including peer and tag labels).
 *
 * If either send or receive MPI datatype size is zero, the function returns
 * immediately and performs no recording.
 *
 * @param _data Context for the intercepted call (contains tool_id used for tracker naming).
 * @param sendcount Number of elements sent.
 * @param sendtype MPI datatype of the send buffer.
 * @param dst Destination rank for the send.
 * @param sendtag MPI tag for the send.
 * @param recvcount Number of elements received.
 * @param recvtype MPI datatype of the receive buffer.
 * @param src Source rank for the receive.
 * @param recvtag MPI tag for the receive.
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int sendcount,
                 MPI_Datatype sendtype, int dst, int sendtag, void*, int recvcount,
                 MPI_Datatype recvtype, int src, int recvtag, MPI_Comm, MPI_Status*)
{
    int _send_size = mpi_type_size(sendtype);
    int _recv_size = mpi_type_size(recvtype);
    if(_send_size == 0 || _recv_size == 0) return;

    if(get_use_perfetto())
    {
        write_perfetto_counter_track<mpi_send>(sendcount * _send_size);
        write_perfetto_counter_track<mpi_recv>(recvcount * _recv_size);
    }

    if(get_use_rocpd())
    {
        rocpd_process_cpu_usage_events<mpi_send>(0, sendcount * _send_size);
        rocpd_process_cpu_usage_events<mpi_recv>(0, recvcount * _send_size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, sendcount * _send_size + recvcount * _recv_size);
        {
            tracker_t _b{ JOIN('/', _name, "send") };
            add(_b, sendcount * _send_size);
            tracker_t _c{ JOIN('/', _name, JOIN('=', "send", dst)) };
            add(_b, sendcount * _send_size);
            add(JOIN('/', _name, "send", JOIN('=', "tag", sendtag)),
                sendcount * _send_size);
            add(JOIN('/', _name, JOIN('=', "send", dst), JOIN('=', "tag", sendtag)),
                sendcount * _send_size);
        }
        {
            tracker_t _b{ JOIN('/', _name, "recv") };
            add(_b, recvcount * _recv_size);
            tracker_t _c{ JOIN('/', _name, JOIN('=', "recv", src)) };
            add(_b, recvcount * _recv_size);
            add(JOIN('/', _name, "recv", JOIN('=', "tag", recvtag)),
                recvcount * _recv_size);
            add(JOIN('/', _name, JOIN('=', "recv", src), JOIN('=', "tag", recvtag)),
                recvcount * _recv_size);
        }
    }
}

// MPI_Gather
/**
 * @brief Audit an MPI_Scatter-like incoming communication event.
 *
 * Records communication sizes for both send and receive sides and emits
 * instrumentation through enabled backends (Perfetto counters, rocpd per-event CPU
 * usage, and timemory trackers).
 *
 * If either MPI datatype has size zero this function returns immediately.
 *
 * @param _data Metadata for the intercepted call (tool identifier is read from `_data.tool_id`).
 * @param audit::incoming Tag indicating this is an incoming/audited call.
 * @param [in]  Send buffer (unused; present to match MPI-like signature).
 * @param sendcount Number of elements sent by each participant.
 * @param sendtype  MPI datatype of the send buffer; its size is used to compute sent bytes.
 * @param [in]  Recv buffer (unused; present to match MPI-like signature).
 * @param recvcount Number of elements received by this participant.
 * @param recvtype  MPI datatype of the receive buffer; its size is used to compute received bytes.
 * @param root      Rank of the root process in the scatter operation (used for labeling timemory trackers).
 * @param MPI_Comm  Communicator (unused by this instrumentation).
 *
 * Side effects:
 * - When Perfetto is enabled, emits counter updates for `mpi_send` and `mpi_recv`.
 * - When rocpd is enabled, records per-event CPU usage samples for send and recv tracks.
 * - When timemory is enabled, updates trackers named by the tool id and additional root/send/recv breakdowns.
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int sendcount,
                 MPI_Datatype sendtype, void*, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm)
{
    int _send_size = mpi_type_size(sendtype);
    int _recv_size = mpi_type_size(recvtype);
    if(_send_size == 0 || _recv_size == 0) return;

    if(get_use_perfetto())
    {
        write_perfetto_counter_track<mpi_send>(sendcount * _send_size);
        write_perfetto_counter_track<mpi_recv>(recvcount * _recv_size);
    }

    if(get_use_rocpd())
    {
        rocpd_process_cpu_usage_events<mpi_send>(0, sendcount * _send_size);
        rocpd_process_cpu_usage_events<mpi_recv>(0, recvcount * _send_size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, sendcount * _send_size + recvcount * _recv_size);
        tracker_t _r(JOIN('/', _name, JOIN('=', "root", root)));
        add(_r, sendcount * _send_size + recvcount * _recv_size);
        add(JOIN('/', _name, JOIN('=', "root", root), "send"), sendcount * _send_size);
        add(JOIN('/', _name, JOIN('=', "root", root), "recv"), recvcount * _recv_size);
    }
}

/**
 * @brief Handle MPI_Alltoall incoming communication for tracking metrics.
 *
 * Computes send/recv byte sizes and, if enabled, records metrics via Perfetto counters,
 * the rocpd data processor, and timemory trackers. No action is taken if either
 * datatype size is zero.
 *
 * @param _data Contains metadata about the intercepted call; `_data.tool_id` is used
 *              as the base name for timemory trackers.
 * @param sendcount Number of elements sent per peer.
 * @param sendtype MPI datatype for send elements (size resolved via `mpi_type_size`).
 * @param recvcount Number of elements received per peer.
 * @param recvtype MPI datatype for receive elements (size resolved via `mpi_type_size`).
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int sendcount,
                 MPI_Datatype sendtype, void*, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm)
{
    int _send_size = mpi_type_size(sendtype);
    int _recv_size = mpi_type_size(recvtype);
    if(_send_size == 0 || _recv_size == 0) return;

    if(get_use_perfetto())
    {
        write_perfetto_counter_track<mpi_send>(sendcount * _send_size);
        write_perfetto_counter_track<mpi_recv>(recvcount * _recv_size);
    }

    if(get_use_rocpd())
    {
        rocpd_process_cpu_usage_events<mpi_send>(0, sendcount * _send_size);
        rocpd_process_cpu_usage_events<mpi_recv>(0, recvcount * _recv_size);
    }

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, sendcount * _send_size + recvcount * _recv_size);
        add(JOIN('/', _name, "send"), sendcount * _send_size);
        add(JOIN('/', _name, "recv"), recvcount * _recv_size);
    }
}
#endif

#if defined(ROCPROFSYS_USE_RCCL)
// Kept for reference, but now gathered throught the SDK callbacks.

/**
 * @brief Handle an incoming NCCL Reduce operation for communication-data tracking.
 *
 * Computes the total transferred bytes from `count` and `datatype` size; if zero, no action is taken.
 * When enabled, emits a Perfetto counter and records a rocpd CPU-usage event for the received bytes.
 * If Timemory is enabled, records the bytes under a tracker named by the Gotcha tool ID and additionally
 * records a tracker annotated with the reduce `root`.
 *
 * @param _data Gotcha wrapper containing metadata; `_data.tool_id` is used as the tracker name.
 * @param count Number of elements reduced; combined with `datatype` to compute total bytes.
 * @param datatype NCCL data type used to determine per-element size.
 * @param root Destination/root rank for the reduction; used to create a root-specific tracker label.
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, const void*,
                 size_t count, ncclDataType_t datatype, ncclRedOp_t, int root, ncclComm_t,
                 hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<rccl_recv>(count * _size);

    if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_recv>(0, count * _size);

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count * _size);
        add(JOIN('/', _name, JOIN('=', "root", root)), count * _size);
    }
}

// ncclSend
// ncclGather
// ncclBcast
/**
 * @brief Handle an incoming RCCL communication event and record data-transfer metrics.
 *
 * Processes an incoming NCCL/RCCL event described by @_data: computes the byte
 * size from the provided `datatype` and `count`, classifies the operation by
 * `_data.tool_id` as a send- or receive-type, and updates enabled backends:
 * - emits Perfetto counter events when Perfetto is enabled,
 * - records per-event CPU-usage samples via rocpd when that path is enabled,
 * - records Timemory trackers and per-peer breakdowns when Timemory is enabled.
 *
 * If the datatype size is non-positive the function returns immediately.
 * If `_data.tool_id` is not recognized as a known send or receive kind, this
 * function throws via ROCPROFSYS_CI_THROW.
 *
 * @param _data Contains metadata for the caught GCC/SDK callback; its `tool_id`
 *              is used to determine the operation kind (send vs recv).
 * @param count Number of elements transferred (multiplied by datatype size to
 *              derive total bytes recorded).
 * @param datatype RCCL datatype used to compute per-element size.
 * @param peer    Peer rank associated with the operation; used for Timemory
 *                per-peer labeling when enabled.
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, size_t count,
                 ncclDataType_t datatype, int peer, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    static auto _send_types = std::unordered_set<std::string>{ "ncclSend", "ncclBcast" };
    static auto _recv_types = std::unordered_set<std::string>{ "ncclGather", "ncclRecv" };

    if(_send_types.count(_data.tool_id) > 0)
    {
        if(get_use_perfetto()) write_perfetto_counter_track<rccl_send>(count * _size);
        if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_send>(0, count * _size);
    }
    else if(_recv_types.count(_data.tool_id) > 0)
    {
        if(get_use_perfetto()) write_perfetto_counter_track<rccl_recv>(count * _size);
        if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_recv>(0, count * _size);
    }
    else
    {
        ROCPROFSYS_CI_THROW(true, "RCCL function not handled: %s", _data.tool_id.c_str());
    }

    if(get_use_perfetto()) write_perfetto_counter_track<rccl_recv>(count * _size);
    if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_recv>(0, count * _size);

    if(rocprofsys::get_use_timemory())
    {
        auto        _name  = std::string_view{ _data.tool_id };
        std::string _label = "root";
        if(_name.find("Send") != std::string::npos) _label = "peer";

        tracker_t _t{ _name };
        add(_t, count * _size);
        add(JOIN('/', _name, JOIN('=', _label, peer)), count * _size);
    }
}

/**
 * @brief Audit handler for an RCCL broadcast operation.
 *
 * If the RCCL data type size is positive, records the total byte count (count * type_size)
 * to enabled backends: Perfetto counters, the rocpd per-event CPU-usage pipeline, and
 * timemory trackers (each backend gated by its corresponding runtime flag).
 *
 * @param _data Contains metadata for the intercepted call; this function uses `_data.tool_id`
 *              as the timemory tracker base name.
 * @param count Number of elements being broadcast.
 * @param datatype RCCL data type of each element; its size (via `rccl_type_size`) is used
 *                 to compute the total byte payload. If `rccl_type_size(datatype) <= 0`,
 *                 the function returns immediately and no instrumentation is emitted.
 * @param root   Rank of the broadcast root; appended to the timemory tracker name when
 *               timemory is enabled.
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, const void*,
                 size_t count, ncclDataType_t datatype, int root, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<rccl_send>(count * _size);
    if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_send>(0, count * _size);

    if(rocprofsys::get_use_timemory())
    {
        auto      _name = std::string_view{ _data.tool_id };
        tracker_t _t{ _name };
        add(_t, count * _size);
        add(JOIN('/', _data.tool_id, JOIN('=', "root", root)), count * _size);
    }
}

// ncclAllReduce
/**
 * @brief Audit handler for incoming ncclReduceScatter-like operations.
 *
 * Determines the byte size of the operation from the NCCL datatype and, based on the
 * recorded tool id in _data, classifies the call as either a send-type (rccl_send) or
 * recv-type (rccl_recv) operation. When applicable this emits:
 *  - a Perfetto counter update,
 *  - a rocpd CPU-usage event,
 *  - and, if enabled, records the byte payload with timemory.
 *
 * If the datatype size is non-positive the call is ignored.
 *
 * @param _data Metadata for the intercepted function; only `_data.tool_id` is consulted
 *              to determine whether the call maps to a send or recv category.
 * @param count Number of elements involved in the operation (used with `datatype`
 *              to compute total bytes).
 * @param datatype NCCL datatype used to compute per-element size.
 *
 * @throws std::runtime_error via ROCPROFSYS_CI_THROW if `_data.tool_id` is unrecognized.
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, const void*,
                 size_t count, ncclDataType_t datatype, ncclRedOp_t, ncclComm_t,
                 hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    static auto _recv_types = std::unordered_set<std::string>{ "ncclAllReduce" };
    static auto _send_types = std::unordered_set<std::string>{ "ncclReduceScatter" };

    if(_send_types.count(_data.tool_id) > 0)
    {
        if(get_use_perfetto()) write_perfetto_counter_track<rccl_send>(count * _size);
        if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_send>(0, count * _size);
    }
    else if(_recv_types.count(_data.tool_id) > 0)
    {
        if(get_use_perfetto()) write_perfetto_counter_track<rccl_recv>(count * _size);
        if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_recv>(0, count * _size);
    }
    else
    {
        ROCPROFSYS_CI_THROW(true, "RCCL function not handled: %s", _data.tool_id.c_str());
    }

    if(rocprofsys::get_use_timemory()) add(_data, count * _size);
}

// ncclAllGather
/**
 * @brief Handle an incoming NCCL All-to-All communication for instrumentation.
 *
 * Emits instrumentation for the total payload size (count * element_size) using
 * any enabled backends: Perfetto counters, the rocpd CPU-usage event path, and
 * timemory aggregation.
 *
 * @param _data Metadata about the intercepted call (source tool/identifier) used
 *              when recording timemory data.
 * @param count Number of elements communicated.
 * @param datatype NCCL data type used to compute element size.
 *
 * @note If the computed element size (from `datatype`) is non-positive, the
 *       function returns immediately and performs no instrumentation.
 */
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, const void*,
                 size_t count, ncclDataType_t datatype, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    if(_size <= 0) return;

    if(get_use_perfetto()) write_perfetto_counter_track<rccl_recv>(count * _size);
    if(get_use_rocpd()) rocpd_process_cpu_usage_events<rccl_recv>(0, count * _size);
    if(rocprofsys::get_use_timemory()) add(_data, count * _size);
}
#endif
}  // namespace component
}  // namespace rocprofsys

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<float, tim::project::rocprofsys>), true, float)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(comm_data, false, void)
