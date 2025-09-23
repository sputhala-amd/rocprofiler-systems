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

#include "library/components/backtrace_metrics.hpp"
#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/node_info.hpp"
#include "core/perfetto.hpp"
#include "core/rocpd/data_processor.hpp"
#include "library/components/ensure_storage.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <timemory/backends/papi.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/components/data_tracker/components.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/components/papi/extern.hpp>
#include <timemory/components/papi/papi_array.hpp>
#include <timemory/components/papi/papi_vector.hpp>
#include <timemory/components/rusage/components.hpp>
#include <timemory/components/rusage/types.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/components/trip_count/extern.hpp>
#include <timemory/macros.hpp>
#include <timemory/math.hpp>
#include <timemory/mpl.hpp>
#include <timemory/mpl/quirks.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/operations.hpp>
#include <timemory/storage.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/variadic.hpp>

#include <array>
#include <cstring>
#include <ctime>
#include <initializer_list>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <pthread.h>
#include <signal.h>

namespace tracing
{
using namespace ::rocprofsys::tracing;
}

namespace rocprofsys
{
namespace component
{
using hw_counters           = typename backtrace_metrics::hw_counters;
using signal_type_instances = thread_data<std::set<int>, category::sampling>;
using backtrace_metrics_init_instances =
    thread_data<backtrace_metrics, category::sampling>;
using sampler_running_instances = thread_data<bool, category::sampling>;
using papi_vector_instances     = thread_data<hw_counters, category::sampling>;
using papi_label_instances = thread_data<std::vector<std::string>, category::sampling>;

namespace
{
struct perfetto_rusage
{};

unique_ptr_t<std::vector<std::string>>&
get_papi_labels(int64_t _tid)
{
    return papi_label_instances::instance(construct_on_thread{ _tid });
}

unique_ptr_t<hw_counters>&
get_papi_vector(int64_t _tid)
{
    return papi_vector_instances::instance(construct_on_thread{ _tid });
}

unique_ptr_t<backtrace_metrics>&
get_backtrace_metrics_init(int64_t _tid)
{
    return backtrace_metrics_init_instances::instance(construct_on_thread{ _tid });
}

unique_ptr_t<bool>&
get_sampler_running(int64_t _tid)
{
    return sampler_running_instances::instance(construct_on_thread{ _tid }, false);
}
}  // namespace

std::string
backtrace_metrics::label()
{
    return "backtrace_metrics";
}

std::string
backtrace_metrics::description()
{
    return "Records sampling data";
}

/**
 * @brief Retrieve hardware counter (PAPI) labels for a thread.
 *
 * Returns the per-thread vector of hardware counter labels associated with the given
 * thread id. If no labels are registered for that thread, an empty vector is returned.
 *
 * @param _tid Thread id used to look up thread-local PAPI labels.
 * @return std::vector<std::string> Vector of hardware counter labels, or empty if none.
 */
std::vector<std::string>
backtrace_metrics::get_hw_counter_labels(int64_t _tid)
{
    auto& _v = get_papi_labels(_tid);
    return (_v) ? *_v : std::vector<std::string>{};
}

/**
 * @brief Returns the global ROCPD data processor instance.
 *
 * Provides access to the singleton rocpd::data_processor used to register
 * categories, threads, tracks, PMCs, events, and samples.
 *
 * @return rocpd::data_processor& Reference to the singleton data processor.
 */
rocpd::data_processor&
get_data_processor()
{
    return rocpd::data_processor::get_instance();
}

/**
 * @brief Start the backtrace metrics sampler.
 *
 * This method is a no-op placeholder kept for interface compatibility.
 */
void
backtrace_metrics::start()
{}

void
backtrace_metrics::stop()
{}

namespace
{
template <typename... Tp>
auto
get_enabled(tim::type_list<Tp...>)
{
    constexpr size_t N  = sizeof...(Tp);
    auto             _v = std::bitset<N>{};
    size_t           _n = 0;
    (_v.set(_n++, trait::runtime_enabled<Tp>::get()), ...);
    return _v;
}
}  // namespace
void
backtrace_metrics::sample(int)
{
    if(!get_enabled(type_list<category::process_sampling, backtrace_metrics>{}).all())
    {
        m_valid.reset();
        return;
    }

    m_valid = get_enabled(categories_t{});

    // return if everything is disabled
    if(!m_valid.any()) return;

    auto _cache = tim::rusage_cache{ RUSAGE_THREAD };
    m_cpu       = tim::get_clock_thread_now<int64_t, std::nano>();
    m_mem_peak  = _cache.get_peak_rss();
    m_ctx_swch  = _cache.get_num_priority_context_switch() +
                 _cache.get_num_voluntary_context_switch();
    m_page_flt = _cache.get_num_major_page_faults() + _cache.get_num_minor_page_faults();

    if constexpr(tim::trait::is_available<hw_counters>::value)
    {
        constexpr auto hw_counters_idx = tim::index_of<hw_counters, categories_t>::value;
        constexpr auto hw_category_idx =
            tim::index_of<category::thread_hardware_counter, categories_t>::value;

        auto _tid = threading::get_id();
        if(m_valid.test(hw_category_idx) && m_valid.test(hw_counters_idx))
        {
            assert(get_papi_vector(_tid).get() != nullptr);
            m_hw_counter = get_papi_vector(_tid)->record();
        }
    }
}

void
backtrace_metrics::configure(bool _setup, int64_t _tid)
{
    auto& _running    = get_sampler_running(_tid);
    bool  _is_running = (!_running) ? false : *_running;

    ensure_storage<comp::trip_count, sampling_wall_clock, sampling_cpu_clock, hw_counters,
                   sampling_percent>{}();

    if(_setup && !_is_running)
    {
        (void) get_debug_sampling();  // make sure query in sampler does not allocate
        assert(_tid == threading::get_id());

        if constexpr(tim::trait::is_available<hw_counters>::value)
        {
            perfetto_counter_track<hw_counters>::init();
            ROCPROFSYS_DEBUG("HW COUNTER: starting...\n");
            if(get_papi_vector(_tid))
            {
                get_papi_vector(_tid)->start();
                *get_papi_labels(_tid) = get_papi_vector(_tid)->get_config()->labels;
            }
        }
    }
    else if(!_setup && _is_running)
    {
        ROCPROFSYS_DEBUG("Destroying sampler for thread %lu...\n", _tid);
        *_running = false;

        if constexpr(tim::trait::is_available<hw_counters>::value)
        {
            if(_tid == threading::get_id())
            {
                if(get_papi_vector(_tid)) get_papi_vector(_tid)->stop();
                ROCPROFSYS_DEBUG("HW COUNTER: stopped...\n");
            }
        }
        ROCPROFSYS_DEBUG("Sampler destroyed for thread %lu\n", _tid);
    }
}

void
backtrace_metrics::init_perfetto(int64_t _tid, valid_array_t _valid)
{
    auto _hw_cnt_labels = *get_papi_labels(_tid);
    auto _tid_name      = JOIN("", '[', _tid, ']');

    if(!perfetto_counter_track<perfetto_rusage>::exists(_tid))
    {
        if(get_valid(category::thread_cpu_time{}, _valid))
            perfetto_counter_track<perfetto_rusage>::emplace(
                _tid, JOIN(' ', "Thread CPU time", _tid_name, "(S)"), "sec");
        if(get_valid(category::thread_peak_memory{}, _valid))
            perfetto_counter_track<perfetto_rusage>::emplace(
                _tid, JOIN(' ', "Thread Peak Memory Usage", _tid_name, "(S)"), "MB");
        if(get_valid(category::thread_context_switch{}, _valid))
            perfetto_counter_track<perfetto_rusage>::emplace(
                _tid, JOIN(' ', "Thread Context Switches", _tid_name, "(S)"));
        if(get_valid(category::thread_page_fault{}, _valid))
            perfetto_counter_track<perfetto_rusage>::emplace(
                _tid, JOIN(' ', "Thread Page Faults", _tid_name, "(S)"));
    }

    if(!perfetto_counter_track<hw_counters>::exists(_tid) &&
       get_valid(type_list<hw_counters>{}, _valid) &&
       get_valid(category::thread_hardware_counter{}, _valid))
    {
        for(auto& itr : _hw_cnt_labels)
        {
            std::string _desc = tim::papi::get_event_info(itr).short_descr;
            if(_desc.empty()) _desc = itr;
            ROCPROFSYS_CI_THROW(_desc.empty(), "Empty description for %s\n", itr.c_str());
            perfetto_counter_track<hw_counters>::emplace(
                _tid, JOIN(' ', "Thread", _desc, _tid_name, "(S)"));
        }
    }
}

/**
 * @brief Finalize Perfetto counters for a thread by emitting end-of-sampling TRACE_COUNTERs.
 *
 * Emits zero-valued TRACE_COUNTER events at the thread stop timestamp for each enabled
 * category in _valid (CPU time, peak memory, context switches, page faults) and for
 * per-thread hardware counters when present. This marks the end of the Perfetto tracks
 * created for the thread so downstream trace consumers can infer final values.
 *
 * @param _tid Thread identifier whose Perfetto tracks will be finalized.
 * @param _valid Bitset of enabled sampling categories; only categories set in this mask
 *               produce TRACE_COUNTER emissions.
 *
 * @throws std::runtime_error if thread information for _tid is missing (via ROCPROFSYS_CI_THROW).
 */
void
backtrace_metrics::fini_perfetto(int64_t _tid, valid_array_t _valid)
{
    auto        _hw_cnt_labels = *get_papi_labels(_tid);
    const auto& _thread_info   = thread_info::get(_tid, SequentTID);

    ROCPROFSYS_CI_THROW(!_thread_info, "Error! missing thread info for tid=%li\n", _tid);
    if(!_thread_info) return;

    uint64_t _ts         = _thread_info->get_stop();
    uint64_t _rusage_idx = 0;

    if(get_valid(category::thread_cpu_time{}, _valid))
    {
        TRACE_COUNTER(trait::name<category::thread_cpu_time>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, 0);
    }

    if(get_valid(category::thread_peak_memory{}, _valid))
    {
        TRACE_COUNTER(trait::name<category::thread_peak_memory>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, 0);
    }

    if(get_valid(category::thread_context_switch{}, _valid))
    {
        TRACE_COUNTER(trait::name<category::thread_context_switch>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, 0);
    }

    if(get_valid(category::thread_page_fault{}, _valid))
    {
        TRACE_COUNTER(trait::name<category::thread_page_fault>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, 0);
    }

    if(get_valid(type_list<hw_counters>{}, _valid) &&
       get_valid(category::thread_hardware_counter{}, _valid))
    {
        for(size_t i = 0; i < perfetto_counter_track<hw_counters>::size(_tid); ++i)
        {
            if(i < _hw_cnt_labels.size())
            {
                TRACE_COUNTER(trait::name<category::thread_hardware_counter>::value,
                              perfetto_counter_track<hw_counters>::at(_tid, i), _ts, 0.0);
            }
        }
    }
}

/**
 * @brief Register ROCpd categories used by backtrace_metrics with the global data processor.
 *
 * Ensures the five thread-level categories (CPU time, peak memory, context switches,
 * page faults, and hardware counters) are inserted into the singleton rocpd::data_processor.
 * This function is safe to call multiple times; registration happens exactly once.
 */
void
rocpd_init_categories()
{
    static bool _is_initialized = false;
    if(_is_initialized) return;

    get_data_processor().insert_category(
        category_enum_id<category::thread_cpu_time>::value,
        trait::name<category::thread_cpu_time>::value);
    get_data_processor().insert_category(
        category_enum_id<category::thread_peak_memory>::value,
        trait::name<category::thread_peak_memory>::value);
    get_data_processor().insert_category(
        category_enum_id<category::thread_context_switch>::value,
        trait::name<category::thread_context_switch>::value);
    get_data_processor().insert_category(
        category_enum_id<category::thread_page_fault>::value,
        trait::name<category::thread_page_fault>::value);
    get_data_processor().insert_category(
        category_enum_id<category::thread_hardware_counter>::value,
        trait::name<category::thread_hardware_counter>::value);

    _is_initialized = true;
}

template <typename Category>
/**
 * @brief Create ROCPD tracks for a thread for the given Category.
 *
 * For non-hardware-counter categories this inserts a single track named
 * "<category_name>_[<tid>]". For Category = category::thread_hardware_counter
 * this creates one track per hardware counter label (PAPI event), using the
 * event's short description (or label if description is empty) and the thread id
 * in the track name.
 *
 * The function registers thread information with the data processor and then
 * inserts the appropriate track(s) referencing that thread.
 *
 * @tparam Category The metric category being initialized. When equal to
 *         `category::thread_hardware_counter` the function enumerates PAPI event
 *         labels and creates a per-counter track.
 * @param _tid Thread identifier used to look up thread_info and to name tracks.
 *
 * @throws std::runtime_error (via ROCPROFSYS_CI_THROW) if a hardware-counter
 *         event has an empty description after attempting to use the PAPI
 *         short description or label.
 */
void
rocpd_init_tracks(int64_t _tid)
{
    auto&       data_processor = get_data_processor();
    auto&       n_info         = node_info::get_instance();
    const auto& t_info         = thread_info::get(_tid, SequentTID);
    auto        _tid_name      = JOIN("", '[', _tid, ']');

    auto thread_idx = data_processor.insert_thread_info(
        n_info.id, getppid(), getpid(), t_info->index_data->system_value,
        JOIN(" ", "Thread", _tid).c_str(), t_info->get_start(), t_info->get_stop(), "{}");

    if constexpr(std::is_same_v<Category, category::thread_hardware_counter>)
    {
        // Initialize hw_counter_tracks and create one track for each hardware counter
        auto _hw_cnt_labels = *get_papi_labels(_tid);
        for(auto& itr : _hw_cnt_labels)
        {
            std::string _desc = tim::papi::get_event_info(itr).short_descr;
            if(_desc.empty()) _desc = itr;
            ROCPROFSYS_CI_THROW(_desc.empty(), "Empty description for %s\n", itr.c_str());

            std::string track_name = JOIN(' ', "Thread", _desc, _tid_name, "(S)");
            data_processor.insert_track(track_name.c_str(), n_info.id, getpid(),
                                        thread_idx, "{}");
        }
    }
    else
        data_processor.insert_track(
            JOIN('_', trait::name<Category>::value, _tid_name).c_str(), n_info.id,
            getpid(), thread_idx, "{}");
}

template <typename Category>
/**
 * @brief Register PMC (performance monitoring counter) descriptions for backtrace metrics with ROCPD.
 *
 * For the given device and thread, inserts PMC description(s) into the global data processor:
 * - If `Category` is `category::thread_hardware_counter`, creates one PMC per hardware counter label
 *   discovered for the thread (uses PAPI event short descriptions as track names). Throws if a
 *   counter has an empty description.
 * - Otherwise, inserts a single PMC description for the category using the category name and the
 *   provided units.
 *
 * @param dev_id Device identifier used to look up the agent/base id for PMC registration.
 * @param units  Units string to attach to the PMC description (e.g., "seconds", "MB").
 * @param _tid   Thread id used to scope track names and to look up per-thread hardware counter labels.
 *
 * @throws std::runtime_error via ROCPROFSYS_CI_THROW if a hardware counter label yields an empty description.
 */
void
rocpd_initialize_backtrace_metrics_pmc(size_t dev_id, const char* units, int64_t _tid)
{
    auto& data_processor = get_data_processor();
    auto  _tid_name      = JOIN("", '[', _tid, ']');

    size_t      EVENT_CODE       = 0;
    size_t      INSTANCE_ID      = 0;
    const char* LONG_DESCRIPTION = "";
    const char* COMPONENT        = "";
    const char* BLOCK            = "";
    const char* EXPRESSION       = "";
    auto        ni               = node_info::get_instance();
    const auto* TARGET_ARCH      = "CPU";

    auto& _agent_manager = agent_manager::get_instance();
    auto  _base_id = _agent_manager.get_agent_by_id(dev_id, agent_type::CPU).base_id;

    if constexpr(std::is_same_v<Category, category::thread_hardware_counter>)
    {
        auto _hw_cnt_labels = *get_papi_labels(_tid);
        for(auto& itr : _hw_cnt_labels)
        {
            std::string _desc = tim::papi::get_event_info(itr).short_descr;
            if(_desc.empty()) _desc = itr;
            ROCPROFSYS_CI_THROW(_desc.empty(), "Empty description for %s\n", itr.c_str());

            std::string track_name = JOIN(' ', "Thread", _desc, _tid_name, "(S)");

            data_processor.insert_pmc_description(
                ni.id, getpid(), _base_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                track_name.c_str(), trait::name<Category>::value,
                trait::name<Category>::description, LONG_DESCRIPTION, COMPONENT, units,
                "ABS", BLOCK, EXPRESSION, 0, 0);
        }
    }
    else
        data_processor.insert_pmc_description(
            ni.id, getpid(), _base_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
            JOIN("_", trait::name<Category>::value, _tid_name).c_str(),
            trait::name<Category>::value, trait::name<Category>::description,
            LONG_DESCRIPTION, COMPONENT, units, "ABS", BLOCK, EXPRESSION, 0, 0);
}

template <typename Category, typename Value>
/**
 * @brief Insert ROCpd PMC events and samples for a backtrace_metrics category and thread.
 *
 * Inserts one or more PMCs and corresponding samples into the global data_processor
 * using the agent base id for the given device. For non-hardware categories a single
 * event/sample is created named "<category>_[tid]". For Category == thread_hardware_counter
 * a per-counter event/sample is created using per-thread PAPI labels and the provided
 * hardware counter values.
 *
 * @tparam Category The metrics category to record (e.g., thread_cpu_time or thread_hardware_counter).
 *
 * @param device_id Device identifier used to resolve the agent/base id (CPU device expected).
 * @param timestamp Timestamp (in the same timebase used by the data_processor) for the sample.
 * @param value Metric value to record. For hardware-counter category this must be
 *              convertible to backtrace_metrics::hw_counter_data_t; for other categories
 *              it should be a numeric value representable as double.
 * @param _tid Thread id associated with the sample; used to construct event/track names.
 *
 * Side effects: inserts events, PMC descriptions and samples into the global data_processor.
 */
void
rocpd_process_backtrace_metrics_events(const uint32_t device_id, uint64_t timestamp,
                                       Value value, int64_t _tid)
{
    auto& data_processor = get_data_processor();
    auto  _tid_name      = JOIN("", '[', _tid, ']');
    auto  event_id =
        data_processor.insert_event(category_enum_id<Category>::value, 0, 0, 0);
    auto& agent_mngr = agent_manager::get_instance();
    auto  base_id    = agent_mngr.get_agent_by_id(device_id, agent_type::CPU).base_id;

    auto insert_event_and_sample = [&](const char* name, double _value) {
        data_processor.insert_pmc_event(event_id, base_id, name, _value);
        data_processor.insert_sample(name, timestamp, event_id);
    };

    if constexpr(std::is_same_v<Category, category::thread_hardware_counter>)
    {
        auto        _hw_cnt_labels = *get_papi_labels(_tid);
        const auto& _hw_counters =
            static_cast<backtrace_metrics::hw_counter_data_t>(value);
        for(size_t i = 0; i < _hw_cnt_labels.size() && i < _hw_counters.size(); ++i)
        {
            std::string _desc = tim::papi::get_event_info(_hw_cnt_labels[i]).short_descr;
            if(_desc.empty()) _desc = _hw_cnt_labels[i];
            std::string track_name = JOIN(' ', "Thread", _desc, _tid_name, "(S)");

            insert_event_and_sample(track_name.c_str(), _hw_counters.at(i));
        }
    }
    else
        insert_event_and_sample(
            JOIN("_", trait::name<Category>::value, _tid_name).c_str(), value);
}

/**
 * @brief Initialize ROCpd categories, tracks, and PMC descriptions for a thread.
 *
 * Ensures ROCpd categories are registered and, for each enabled metric in _valid,
 * creates per-thread tracks and performance-monitoring-counter (PMC) descriptions
 * so subsequent samples for the given thread id can be recorded.
 *
 * @param _tid Thread identifier used when creating thread-specific tracks and PMCs.
 * @param _valid Bitset indicating which metric categories (CPU time, peak memory,
 *               context switches, page faults, hardware counters, etc.) should be
 *               initialized for this thread.
 */
void
backtrace_metrics::init_rocpd(int64_t _tid, valid_array_t _valid)
{
    rocpd_init_categories();
    if(get_valid(category::thread_cpu_time{}, _valid))
    {
        rocpd_init_tracks<category::thread_cpu_time>(_tid);
        rocpd_initialize_backtrace_metrics_pmc<category::thread_cpu_time>(0, "sec", _tid);
    }
    if(get_valid(category::thread_peak_memory{}, _valid))
    {
        rocpd_init_tracks<category::thread_peak_memory>(_tid);
        rocpd_initialize_backtrace_metrics_pmc<category::thread_peak_memory>(0, "MB",
                                                                             _tid);
    }
    if(get_valid(category::thread_context_switch{}, _valid))
    {
        rocpd_init_tracks<category::thread_context_switch>(_tid);
        rocpd_initialize_backtrace_metrics_pmc<category::thread_context_switch>(0, "",
                                                                                _tid);
    }
    if(get_valid(category::thread_page_fault{}, _valid))
    {
        rocpd_init_tracks<category::thread_page_fault>(_tid);
        rocpd_initialize_backtrace_metrics_pmc<category::thread_page_fault>(0, "", _tid);
    }
    if(get_valid(type_list<hw_counters>{}, _valid) &&
       get_valid(category::thread_hardware_counter{}, _valid))
    {
        rocpd_init_tracks<category::thread_hardware_counter>(_tid);
        rocpd_initialize_backtrace_metrics_pmc<category::thread_hardware_counter>(0, "",
                                                                                  _tid);
    }
}

/**
 * @brief Finalize and emit ROCpd events for a thread's backtrace metrics.
 *
 * Emits final ROCpd events (with zeroed values) at the thread's stop timestamp
 * for each enabled category present in _valid. Handles CPU time, peak memory,
 * context switches, page faults, and per-counter hardware counters.
 *
 * This function requires valid thread info for _tid and will throw if that
 * information is missing.
 *
 * @param _tid Thread identifier whose ROCpd tracks are being finalized.
 * @param _valid Bitset describing which categories/type-lists are enabled.
 *
 * @throws std::runtime_error via ROCPROFSYS_CI_THROW if thread info for _tid
 *         cannot be obtained.
 */
void
backtrace_metrics::fini_rocpd(int64_t _tid, valid_array_t _valid)
{
    const auto& _thread_info = thread_info::get(_tid, SequentTID);

    ROCPROFSYS_CI_THROW(!_thread_info, "Error! missing thread info for tid=%li\n", _tid);
    if(!_thread_info) return;

    uint64_t _ts = _thread_info->get_stop();

    if(get_valid(category::thread_cpu_time{}, _valid))
    {
        rocpd_process_backtrace_metrics_events<category::thread_cpu_time, double>(
            0, _ts, 0, _tid);
    }

    if(get_valid(category::thread_peak_memory{}, _valid))
    {
        rocpd_process_backtrace_metrics_events<category::thread_peak_memory, double>(
            0, _ts, 0, _tid);
    }

    if(get_valid(category::thread_context_switch{}, _valid))
    {
        rocpd_process_backtrace_metrics_events<category::thread_context_switch, int64_t>(
            0, _ts, 0, _tid);
    }

    if(get_valid(category::thread_page_fault{}, _valid))
    {
        rocpd_process_backtrace_metrics_events<category::thread_page_fault, int64_t>(
            0, _ts, 0, _tid);
    }

    if(get_valid(type_list<hw_counters>{}, _valid) &&
       get_valid(category::thread_hardware_counter{}, _valid))
    {
        auto              _hw_cnt_labels = *get_papi_labels(_tid);
        hw_counter_data_t zero_counters{};
        zero_counters.fill(0.0);

        rocpd_process_backtrace_metrics_events<category::thread_hardware_counter,
                                               hw_counter_data_t>(0, _ts, zero_counters,
                                                                  _tid);
    }
}

/**
 * @brief Subtracts metrics from another backtrace_metrics into this instance.
 *
 * Subtracts each enabled metric field of @_rhs from the corresponding field in
 * this object. Only categories that are enabled for this instance are updated:
 * - thread_cpu_time: subtracts m_cpu
 * - thread_peak_memory: subtracts m_mem_peak
 * - thread_context_switch: subtracts m_ctx_swch
 * - thread_page_fault: subtracts m_page_flt
 * - thread_hardware_counter: performs element-wise subtraction across m_hw_counter
 *
 * @param _rhs Source metrics to subtract.
 * @return backtrace_metrics& Reference to this instance after subtraction.
 */
backtrace_metrics&
backtrace_metrics::operator-=(const backtrace_metrics& _rhs)
{
    auto& _lhs = *this;

    if(_lhs(category::thread_cpu_time{}))
    {
        _lhs.m_cpu -= _rhs.m_cpu;
    }

    if(_lhs(category::thread_peak_memory{}))
    {
        _lhs.m_mem_peak -= _rhs.m_mem_peak;
    }

    if(_lhs(category::thread_context_switch{}))
    {
        _lhs.m_ctx_swch -= _rhs.m_ctx_swch;
    }

    if(_lhs(category::thread_page_fault{}))
    {
        _lhs.m_page_flt -= _rhs.m_page_flt;
    }

    if(_lhs(type_list<hw_counters>{}) && _lhs(category::thread_hardware_counter{}))
    {
        for(size_t i = 0; i < _lhs.m_hw_counter.size(); ++i)
            _lhs.m_hw_counter.at(i) -= _rhs.m_hw_counter.at(i);
    }

    return _lhs;
}

/**
 * @brief Emit Perfetto trace counters for this backtrace_metrics sample.
 *
 * When categories are enabled, writes TRACE_COUNTER events to the Perfetto
 * tracks associated with the given thread:
 * - thread_cpu_time: emits CPU time in seconds,
 * - thread_peak_memory: emits peak RSS in megabytes,
 * - thread_context_switch: emits context-switch count,
 * - thread_page_fault: emits page-fault count,
 * - thread_hardware_counter: emits each hardware counter value on its per-counter track.
 *
 * @param _tid Thread identifier used to select per-thread Perfetto tracks.
 * @param _ts  Timestamp to associate with emitted TRACE_COUNTER events (Perfetto timestamp units).
 */
void
backtrace_metrics::post_process_perfetto(int64_t _tid, uint64_t _ts) const
{
    uint64_t _rusage_idx = 0;

    if((*this)(category::thread_cpu_time{}))
    {
        TRACE_COUNTER(trait::name<category::thread_cpu_time>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, m_cpu / units::sec);
    }

    if((*this)(category::thread_peak_memory{}))
    {
        TRACE_COUNTER(trait::name<category::thread_peak_memory>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, m_mem_peak / units::megabyte);
    }

    if((*this)(category::thread_context_switch{}))
    {
        TRACE_COUNTER(trait::name<category::thread_context_switch>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, m_ctx_swch);
    }

    if((*this)(category::thread_page_fault{}))
    {
        TRACE_COUNTER(trait::name<category::thread_page_fault>::value,
                      perfetto_counter_track<perfetto_rusage>::at(_tid, _rusage_idx++),
                      _ts, m_page_flt);
    }

    if((*this)(type_list<hw_counters>{}) && (*this)(category::thread_hardware_counter{}))
    {
        for(size_t i = 0; i < perfetto_counter_track<hw_counters>::size(_tid); ++i)
        {
            if(i < m_hw_counter.size())
            {
                TRACE_COUNTER(trait::name<category::thread_hardware_counter>::value,
                              perfetto_counter_track<hw_counters>::at(_tid, i), _ts,
                              m_hw_counter.at(i));
            }
        }
    }
}

/**
 * @brief Emit ROCpd events/samples for all enabled backtrace metrics for a thread.
 *
 * For each category enabled on this backtrace_metrics instance, this function
 * inserts an event and associated sample into the ROCpd data processor using
 * the stored metric values. Values are converted to the units expected by the
 * corresponding ROCpd PMCs (seconds for CPU time, megabytes for peak memory,
 * raw counts for context switches/page faults, and per-counter data for
 * hardware counters).
 *
 * Only categories that evaluate true when invoked as a predicate on *this are
 * emitted. Hardware counter data is emitted as a single hw_counter_data_t blob
 * when both hardware counters and the thread_hardware_counter category are enabled.
 *
 * @param _tid Thread identifier used to resolve the target tracks in ROCpd.
 * @param _ts  Timestamp forwarded to the data processor and used for inserted samples.
 */
void
backtrace_metrics::post_process_rocpd(int64_t _tid, uint64_t _ts) const
{
    auto is_category_enabled = [&](const auto& _category) { return (*this)(_category); };

    if(is_category_enabled(category::thread_cpu_time{}))
    {
        rocpd_process_backtrace_metrics_events<category::thread_cpu_time, double>(
            0, _ts, m_cpu / units::sec, _tid);
    }

    if(is_category_enabled(category::thread_peak_memory{}))
    {
        rocpd_process_backtrace_metrics_events<category::thread_peak_memory, double>(
            0, _ts, m_mem_peak / units::megabyte, _tid);
    }

    if(is_category_enabled(category::thread_context_switch{}))
    {
        rocpd_process_backtrace_metrics_events<category::thread_context_switch, int64_t>(
            0, _ts, m_ctx_swch, _tid);
    }

    if(is_category_enabled(category::thread_page_fault{}))
    {
        rocpd_process_backtrace_metrics_events<category::thread_page_fault, int64_t>(
            0, _ts, m_page_flt, _tid);
    }
    if(is_category_enabled(type_list<hw_counters>{}) &&
       is_category_enabled(category::thread_hardware_counter{}))
    {
        rocpd_process_backtrace_metrics_events<category::thread_hardware_counter,
                                               hw_counter_data_t>(0, _ts, m_hw_counter,
                                                                  _tid);
    }
}
}  // namespace component
}  // namespace rocprofsys

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_wall_clock>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_cpu_clock>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_fraction>), true,
    double)

TIMEMORY_INITIALIZE_STORAGE(rocprofsys::component::backtrace_metrics)
