// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#include "core/agent.hpp"
#if defined(NDEBUG)
#    undef NDEBUG
#endif

#include "core/agent_manager.hpp"
#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/gpu.hpp"
#include "core/node_info.hpp"
#include "core/perfetto.hpp"
#include "core/rocpd/data_processor.hpp"
#include "core/state.hpp"
#include "library/amd_smi.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/locking.hpp>

#include <cassert>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

#define ROCPROFSYS_AMD_SMI_CALL(...)                                                     \
    ::rocprofsys::amd_smi::check_error(__FILE__, __LINE__, __VA_ARGS__)

namespace rocprofsys
{
namespace amd_smi
{
using bundle_t          = std::deque<data>;
using sampler_instances = thread_data<bundle_t, category::amd_smi>;

namespace
{
/**
 * @brief Return a stable numeric identifier for the current thread.
 *
 * Returns a thread-local, cached thread identifier suitable for use as a lightweight
 * per-thread key (e.g., in maps or logs). The value is stable for the lifetime of
 * the thread and is retrieved once per thread and cached thereafter.
 *
 * @return int64_t Numeric thread identifier for the calling thread.
 */
int64_t
get_tid()
{
    static thread_local auto _v = threading::get_id();
    return _v;
}

/**
 * @brief Access the global ROCPD data processor singleton.
 *
 * Returns a reference to the single, process-wide rocpd::data_processor instance
 * used to register categories, tracks, and emit PMC data.
 *
 * @return rocpd::data_processor& Reference to the singleton data processor.
 */
rocpd::data_processor&
get_data_processor()
{
    return rocpd::data_processor::get_instance();
}

/**
 * @brief Register the AMD SMI category with the global rocpd data processor.
 *
 * Inserts the AMD SMI category identifier and its human-readable name into the
 * singleton data_processor so downstream ROCPD tracks and PMCs can be created
 * or looked up by category.
 */
void
rocpd_initialize_category()
{
    get_data_processor().insert_category(ROCPROFSYS_CATEGORY_AMD_SMI,
                                         trait::name<category::amd_smi>::value);
}

/**
 * @brief Register AMD SMI metric tracks with the global ROCPD data processor.
 *
 * Inserts per-category tracks (MM busy, power, temperature, memory usage)
 * into the singleton data_processor using the current node id and process id.
 * The tracks are created without a specific thread id (internal AMD-SMI sampling).
 */
void
rocpd_initialize_smi_tracks()
{
    auto&      data_processor = get_data_processor();
    auto&      n_info         = node_info::get_instance();
    const auto thread_id      = std::nullopt;  // Internal thread ID for amd-smi

    data_processor.insert_track(trait::name<category::amd_smi_mm_busy>::value, n_info.id,
                                getpid(), thread_id);
    data_processor.insert_track(trait::name<category::amd_smi_power>::value, n_info.id,
                                getpid(), thread_id);
    data_processor.insert_track(trait::name<category::amd_smi_temp>::value, n_info.id,
                                getpid(), thread_id);
    data_processor.insert_track(trait::name<category::amd_smi_memory_usage>::value,
                                n_info.id, getpid(), thread_id);
}

/**
 * @brief Register ROC Profiler PMC descriptions for AMD SMI metrics of a GPU.
 *
 * Inserts PMC descriptions into the global rocpd::data_processor for the
 * specified GPU so downstream ROCPD consumers can interpret samples for:
 * - memory-mapped (MM) busy percentage,
 * - temperature (°C),
 * - power (W),
 * - memory usage (MB).
 *
 * @param gpu_id Identifier for the GPU whose PMC descriptions should be created.
 *               This is used to look up the agent's base_id and associate the
 *               inserted PMC descriptions with that GPU.
 */
void
rocpd_initialize_smi_pmc(size_t gpu_id)
{
    auto& data_processor = get_data_processor();
    // find the proper values for a following definitions
    size_t      EVENT_CODE       = 0;
    size_t      INSTANCE_ID      = 0;
    const char* LONG_DESCRIPTION = "";
    const char* COMPONENT        = "";
    const char* BLOCK            = "";
    const char* EXPRESSION       = "";
    const char* CELSIUS_DEGREES  = "\u00B0C";
    auto        ni               = node_info::get_instance();
    const auto* TARGET_ARCH      = "GPU";

    auto& _agent_manager = agent_manager::get_instance();
    auto  base_id = _agent_manager.get_agent_by_id(gpu_id, agent_type::GPU).base_id;

    data_processor.insert_pmc_description(
        ni.id, getpid(), base_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
        trait::name<category::amd_smi_mm_busy>::value, "Busy",
        trait::name<category::amd_smi_mm_busy>::description, LONG_DESCRIPTION, COMPONENT,
        "%", "ABS", BLOCK, EXPRESSION, 0, 0);

    data_processor.insert_pmc_description(
        ni.id, getpid(), base_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
        trait::name<category::amd_smi_temp>::value, "Temp",
        trait::name<category::amd_smi_temp>::description, LONG_DESCRIPTION, COMPONENT,
        CELSIUS_DEGREES, "ABS", BLOCK, EXPRESSION, 0, 0);

    data_processor.insert_pmc_description(
        ni.id, getpid(), base_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
        trait::name<category::amd_smi_power>::value, "Pow",
        trait::name<category::amd_smi_power>::description, LONG_DESCRIPTION, COMPONENT,
        "w", "ABS", BLOCK, EXPRESSION, 0, 0);

    data_processor.insert_pmc_description(
        ni.id, getpid(), base_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
        trait::name<category::amd_smi_memory_usage>::value, "MemUsg",
        trait::name<category::amd_smi_memory_usage>::description, LONG_DESCRIPTION,
        COMPONENT, "MB", "ABS", BLOCK, EXPRESSION, 0, 0);
}

/**
 * @brief Emit ROCPD PMC events and samples for AMD SMI metrics for a given GPU.
 *
 * Conditionally inserts PMC event(s) and corresponding samples into the global
 * rocpd data processor for the enabled metrics in @p settings (GPU busy,
 * temperature, power, memory usage). Uses the GPU agent's base ID to attribute
 * emitted PMCs. If none of the metrics are enabled this function is a no-op.
 *
 * @param device_id GPU device identifier used to look up the agent/base ID.
 * @param settings Per-device AMD SMI settings that indicate which metrics to emit.
 * @param timestamp Timestamp associated with the samples.
 * @param busy GPU busy percentage value.
 * @param temp GPU temperature value.
 * @param power GPU power value.
 * @param usage GPU memory usage value.
 */
void
rocpd_process_smi_pmc_events(const uint32_t device_id, const amd_smi::settings& settings,
                             uint64_t timestamp, double busy, double temp, double power,
                             double usage)
{
    if(!(settings.busy || settings.temp || settings.power || settings.mem_usage)) return;

    auto& data_processor = get_data_processor();
    auto  event_id = data_processor.insert_event(ROCPROFSYS_CATEGORY_AMD_SMI, 0, 0, 0);

    auto& _agent_manager = agent_manager::get_instance();
    auto  base_id = _agent_manager.get_agent_by_id(device_id, agent_type::GPU).base_id;

    auto insert_event_and_sample = [&](bool enabled, const char* name, double value) {
        if(!enabled) return;
        data_processor.insert_pmc_event(event_id, base_id, name, value);
        data_processor.insert_sample(name, timestamp, event_id);
    };

    insert_event_and_sample(settings.busy, trait::name<category::amd_smi_mm_busy>::value,
                            busy);
    insert_event_and_sample(settings.temp, trait::name<category::amd_smi_temp>::value,
                            temp);
    insert_event_and_sample(settings.power, trait::name<category::amd_smi_power>::value,
                            power);
    insert_event_and_sample(settings.mem_usage,
                            trait::name<category::amd_smi_memory_usage>::value, usage);
}

/**
 * @brief Retrieve the mutable AMD SMI settings for a given device.
 *
 * Returns a reference to the settings entry for device `_dev_id` in an internal
 * map, creating a default-constructed settings object if none exists.
 *
 * @param _dev_id Device identifier used as the key for the settings lookup.
 * @return amd_smi::settings& Reference to the settings for the specified device.
 */
auto&
get_settings(uint32_t _dev_id)
{
    static auto _v = std::unordered_map<uint32_t, amd_smi::settings>{};
    return _v[_dev_id];
}

bool&
is_initialized()
{
    static bool _v = false;
    return _v;
}

amdsmi_version_t&
get_version()
{
    static amdsmi_version_t _v = {};

    if(_v.major == 0 && _v.minor == 0)
    {
        auto _err = amdsmi_get_lib_version(&_v);
        if(_err != AMDSMI_STATUS_SUCCESS)
            ROCPROFSYS_THROW(
                "amdsmi_get_version failed. No version information available.");
    }

    return _v;
}

void
check_error(const char* _file, int _line, amdsmi_status_t _code, bool* _option = nullptr)
{
    if(_code == AMDSMI_STATUS_SUCCESS)
        return;
    else if(_code == AMDSMI_STATUS_NOT_SUPPORTED && _option)
    {
        *_option = false;
        return;
    }

    const char* _msg = nullptr;
    auto        _err = amdsmi_status_code_to_string(_code, &_msg);
    if(_err != AMDSMI_STATUS_SUCCESS)
        ROCPROFSYS_THROW(
            "amdsmi_status_code_to_string failed. No error message available. "
            "Error code %i originated at %s:%i\n",
            static_cast<int>(_code), _file, _line);
    ROCPROFSYS_THROW("[%s:%i] Error code %i :: %s", _file, _line, static_cast<int>(_code),
                     _msg);
}

std::atomic<State>&
get_state()
{
    static std::atomic<State> _v{ State::PreInit };
    return _v;
}
}  // namespace

//--------------------------------------------------------------------------------------//

size_t                           data::device_count     = 0;
std::set<uint32_t>               data::device_list      = {};
std::unique_ptr<data::promise_t> data::polling_finished = {};

data::data(uint32_t _dev_id) { sample(_dev_id); }

/**
 * @brief Collects AMD SMI metrics for a single GPU device and stores them in this instance.
 *
 * Samples configured AMD SMI metrics for device `_dev_id` (timestamped) and populates
 * instance members such as m_busy_perc, m_temp, m_power, m_mem_usage and m_xcp_metrics.
 * Sampling is skipped if running in a child process or if the global AMD SMI state is not
 * Active. Individual AMD SMI read failures are caught; on such an error sampling is
 * disabled globally (state set to Disabled) to avoid repeated failures.
 *
 * @param _dev_id GPU device index to sample.
 *
 * @note This function updates internal state (member fields) and may mutate the global
 *       AMD SMI sampling state on error. It does not throw; AMD SMI errors are handled
 *       internally by disabling future sampling.
 */
void
data::sample(uint32_t _dev_id)
{
    if(is_child_process()) return;

    auto _ts = tim::get_clock_real_now<size_t, std::nano>();
    assert(_ts < std::numeric_limits<int64_t>::max());
    amdsmi_gpu_metrics_t _gpu_metrics;
    bool                 _vcn_or_jpeg_activity_enabled = false;

    auto _state = get_state().load();

    if(_state != State::Active) return;

    m_dev_id = _dev_id;
    m_ts     = _ts;

#define ROCPROFSYS_AMDSMI_GET(OPTION, FUNCTION, ...)                                     \
    if(OPTION)                                                                           \
    {                                                                                    \
        try                                                                              \
        {                                                                                \
            ROCPROFSYS_AMD_SMI_CALL(FUNCTION(__VA_ARGS__), &OPTION);                     \
        } catch(std::runtime_error & _e)                                                 \
        {                                                                                \
            ROCPROFSYS_VERBOSE_F(                                                        \
                0, "[%s] Exception: %s. Disabling future samples from amd-smi...\n",     \
                #FUNCTION, _e.what());                                                   \
            get_state().store(State::Disabled);                                          \
        }                                                                                \
    }

    amdsmi_processor_handle sample_handle = gpu::get_handle_from_id(_dev_id);
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).busy, amdsmi_get_gpu_activity,
                          sample_handle, &m_busy_perc);
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).temp, amdsmi_get_temp_metric,
                          sample_handle, AMDSMI_TEMPERATURE_TYPE_JUNCTION,
                          AMDSMI_TEMP_CURRENT, &m_temp);
#if(AMDSMI_LIB_VERSION_MAJOR == 2 && AMDSMI_LIB_VERSION_MINOR == 0) ||                   \
    (AMDSMI_LIB_VERSION_MAJOR == 25 && AMDSMI_LIB_VERSION_MINOR == 2)
    // This was a transient change in the AMD SMI API. It was never officially released.
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).power, amdsmi_get_power_info,
                          sample_handle, 0, &m_power)
#else
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).power, amdsmi_get_power_info,
                          sample_handle, &m_power)
#endif
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).mem_usage, amdsmi_get_gpu_memory_usage,
                          sample_handle, AMDSMI_MEM_TYPE_VRAM, &m_mem_usage);
    _vcn_or_jpeg_activity_enabled =
        get_settings(m_dev_id).vcn_activity || get_settings(m_dev_id).jpeg_activity;
    ROCPROFSYS_AMDSMI_GET(_vcn_or_jpeg_activity_enabled, amdsmi_get_gpu_metrics_info,
                          sample_handle, &_gpu_metrics);

    // Process metrics if either VCN or JPEG activity is enabled
    if(_vcn_or_jpeg_activity_enabled)
    {
        // Helper lambda to fill busy metrics from a source array
        auto fill_busy_metrics = [](auto& dest, const auto& src) {
            for(const auto& val : src)
            {
                if(val != UINT16_MAX) dest.push_back(val);
            }
        };

        if(gpu::is_vcn_activity_supported(m_dev_id) &&
           gpu::is_jpeg_activity_supported(m_dev_id))
        {
            // Both VCN and JPEG are supported - create one entry with both metrics
            xcp_metrics_t metrics;
            fill_busy_metrics(metrics.vcn_busy, _gpu_metrics.vcn_activity);
            fill_busy_metrics(metrics.jpeg_busy, _gpu_metrics.jpeg_activity);
            if(!metrics.vcn_busy.empty() || !metrics.jpeg_busy.empty())
                m_xcp_metrics.push_back(metrics);
        }
        else if(gpu::is_vcn_activity_supported(m_dev_id))
        {
            // Only VCN is supported
            xcp_metrics_t metrics;
            fill_busy_metrics(metrics.vcn_busy, _gpu_metrics.vcn_activity);
            if(!metrics.vcn_busy.empty()) m_xcp_metrics.push_back(metrics);
        }
        else if(gpu::is_jpeg_activity_supported(m_dev_id))
        {
            // Only JPEG is supported
            xcp_metrics_t metrics;
            fill_busy_metrics(metrics.jpeg_busy, _gpu_metrics.jpeg_activity);
            if(!metrics.jpeg_busy.empty()) m_xcp_metrics.push_back(metrics);
        }
        else
        {
            // Neither is supported - use XCP stats
            // Each XCP gets one entry with both its VCN and JPEG metrics
            for(const auto& xcp : _gpu_metrics.xcp_stats)
            {
                xcp_metrics_t metrics;
                fill_busy_metrics(metrics.vcn_busy, xcp.vcn_busy);
                fill_busy_metrics(metrics.jpeg_busy, xcp.jpeg_busy);
                if(!metrics.vcn_busy.empty() || !metrics.jpeg_busy.empty())
                    m_xcp_metrics.push_back(metrics);
            }
        }
    }

#undef ROCPROFSYS_AMDSMI_GET
}

void
data::print(std::ostream& _os) const
{
    std::stringstream _ss{};

#if ROCPROFSYS_USE_ROCM > 0
    _ss << "device: " << m_dev_id << ", gpu busy: = " << m_busy_perc.gfx_activity
        << "%, mm busy: = " << m_busy_perc.mm_activity
        << "%, umc busy: = " << m_busy_perc.umc_activity << "%, temp = " << m_temp
        << ", current power = " << m_power.current_socket_power
        << ", memory usage = " << m_mem_usage;
#endif
    _os << _ss.str();
}

namespace
{
std::vector<unique_ptr_t<bundle_t>*> _bundle_data{};
}

/**
 * @brief Configure per-device sampling bundles and initialize integrations.
 *
 * Ensures internal bundle storage is sized for the detected GPU count, attaches
 * per-device sampler bundles for enabled devices (creating empty bundles when
 * needed), collects an initial sample for each enabled device, and initializes
 * ROC Perf Data (rocpd) category and SMI tracks when rocpd usage is enabled.
 *
 * Side effects:
 * - Resizes and populates _bundle_data.
 * - May allocate new bundle_t instances for devices in data::device_list.
 * - Calls data::get_initial().sample(...) once per enabled device.
 * - Calls rocpd_initialize_category() and rocpd_initialize_smi_tracks() if
 *   get_use_rocpd() returns true.
 */
void
config()
{
    _bundle_data.resize(data::device_count, nullptr);
    for(size_t i = 0; i < data::device_count; ++i)
    {
        if(data::device_list.count(i) > 0)
        {
            _bundle_data.at(i) = &sampler_instances::get()->at(i);
            if(!*_bundle_data.at(i))
                *_bundle_data.at(i) = unique_ptr_t<bundle_t>{ new bundle_t{} };
        }
    }
    data::get_initial().resize(data::device_count);
    for(auto itr : data::device_list)
        data::get_initial().at(itr).sample(itr);

    if(get_use_rocpd())
    {
        rocpd_initialize_category();
        rocpd_initialize_smi_tracks();
    }
}

/**
 * @brief Polls enabled AMD SMI devices and records a new sample for each.
 *
 * Acquires the AMD SMI category mutex, iterates the configured device list, and
 * for each device in the Active sampling state appends a freshly constructed
 * data sample to that device's bundle. Devices without an allocated bundle or
 * when sampling is not Active are skipped.
 */
void
sample()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    for(auto itr : data::device_list)
    {
        if(amd_smi::get_state() != State::Active) continue;
        ROCPROFSYS_DEBUG_F("Polling amd-smi for device %u...\n", itr);
        auto& _data = *_bundle_data.at(itr);
        if(!_data) continue;
        _data->emplace_back(data{ itr });
        ROCPROFSYS_DEBUG_F("    %s\n", TIMEMORY_JOIN("", _data->back()).c_str());
    }
}

void
set_state(State _v)
{
    amd_smi::get_state().store(_v);
}

std::vector<data>&
data::get_initial()
{
    static std::vector<data> _v{};
    return _v;
}

bool
data::setup()
{
    perfetto_counter_track<data>::init();
    amd_smi::set_state(State::PreInit);
    return true;
}

bool
data::shutdown()
{
    amd_smi::set_state(State::Finalized);
    return true;
}

#define GPU_METRIC(COMPONENT, ...)                                                       \
    if constexpr(tim::trait::is_available<COMPONENT>::value)                             \
    {                                                                                    \
        auto* _val = _v.get<COMPONENT>();                                                \
        if(_val)                                                                         \
        {                                                                                \
            _val->set_value(itr.__VA_ARGS__);                                            \
            _val->set_accum(itr.__VA_ARGS__);                                            \
        }                                                                                \
    }

/**
 * @brief Post-processes collected AMD SMI samples for a single device.
 *
 * Processes the per-thread AMD SMI sample bundle for device `_dev_id`, converting
 * collected metrics into perfetto counter tracks and/or ROC Profiler PMC events
 * according to current settings. This will initialize perfetto counter tracks
 * and/or ROCPD PMC descriptions on demand and emit counter samples for GPU
 * busy, temperature, power, memory usage, and per-XCP VCN/JPEG activity when
 * enabled.
 *
 * If `_dev_id` is out of range, or required thread timing information is
 * missing/invalid for a sample, that sample is skipped. When ROCPD usage is
 * enabled, SMI PMC descriptions/tracks will be initialized before emitting PMC
 * events.
 *
 * @param _dev_id GPU device index for which to post-process samples.
 */
void
data::post_process(uint32_t _dev_id)
{
    using component::sampling_gpu_busy_gfx;
    using component::sampling_gpu_busy_mm;
    using component::sampling_gpu_busy_umc;
    using component::sampling_gpu_jpeg;
    using component::sampling_gpu_memory;
    using component::sampling_gpu_power;
    using component::sampling_gpu_temp;
    using component::sampling_gpu_vcn;

    if(device_count < _dev_id) return;

    auto&       _amd_smi_v   = sampler_instances::get()->at(_dev_id);
    auto        _amd_smi     = (_amd_smi_v) ? *_amd_smi_v : std::deque<amd_smi::data>{};
    const auto& _thread_info = thread_info::get(0, InternalTID);

    ROCPROFSYS_VERBOSE(1, "Post-processing %zu amd-smi samples from device %u\n",
                       _amd_smi.size(), _dev_id);

    ROCPROFSYS_CI_THROW(!_thread_info, "Missing thread info for thread 0");
    if(!_thread_info) return;

    auto _settings = get_settings(_dev_id);

    auto use_perfetto = get_use_perfetto();
    auto use_rocpd    = get_use_rocpd();

    if(use_rocpd)
    {
        rocpd_initialize_smi_pmc(_dev_id);
    }

    for(auto& itr : _amd_smi)
    {
        using counter_track = perfetto_counter_track<data>;
        if(itr.m_dev_id != _dev_id) continue;

        uint64_t _ts = itr.m_ts;
        if(!_thread_info->is_valid_time(_ts)) continue;

        double _gfxbusy = itr.m_busy_perc.gfx_activity;
        double _umcbusy = itr.m_busy_perc.umc_activity;
        double _mmbusy  = itr.m_busy_perc.mm_activity;
        double _temp    = itr.m_temp;
        double _power   = itr.m_power.current_socket_power;
        double _usage   = itr.m_mem_usage / static_cast<double>(units::megabyte);

        auto setup_perfetto_counter_tracks = [&]() {
            if(counter_track::exists(_dev_id)) return;

            auto addendum = [&](const char* _v) {
                return JOIN(" ", "GPU", _v, JOIN("", '[', _dev_id, ']'), "(S)");
            };

            auto addendum_blk = [&](std::size_t _i, const char* _metric,
                                    std::size_t xcp_idx = SIZE_MAX) {
                if(xcp_idx != SIZE_MAX)
                {
                    return JOIN(
                        " ", "GPU", JOIN("", '[', _dev_id, ']'), _metric,
                        JOIN("", "XCP_", xcp_idx, ": [", (_i < 10 ? "0" : ""), _i, ']'),
                        "(S)");
                }
                else
                {
                    return JOIN(" ", "GPU", JOIN("", '[', _dev_id, ']'), _metric,
                                JOIN("", "[", (_i < 10 ? "0" : ""), _i, ']'), "(S)");
                }
            };

            if(_settings.busy)
            {
                counter_track::emplace(_dev_id, addendum("GFX Busy"), "%");
                counter_track::emplace(_dev_id, addendum("UMC Busy"), "%");
                counter_track::emplace(_dev_id, addendum("MM Busy"), "%");
            }
            if(_settings.temp)
            {
                counter_track::emplace(_dev_id, addendum("Temperature"), "deg C");
            }
            if(_settings.power)
            {
                counter_track::emplace(_dev_id, addendum("Current Power"), "watts");
            }
            if(_settings.mem_usage)
            {
                counter_track::emplace(_dev_id, addendum("Memory Usage"), "megabytes");
            }
            if(_settings.vcn_activity)
            {
                if(itr.m_xcp_metrics.empty())
                {
                    ROCPROFSYS_VERBOSE(
                        1, "No VCN activity data collected from device %u\n", _dev_id);
                }
                else if(gpu::is_vcn_activity_supported(_dev_id))
                {
                    // For VCN activity, use simple indexing
                    for(std::size_t i = 0; i < std::size(itr.m_xcp_metrics[0].vcn_busy);
                        ++i)
                        counter_track::emplace(_dev_id, addendum_blk(i, "VCN Activity"),
                                               "%");
                }
                else
                {
                    for(std::size_t xcp = 0; xcp < std::size(itr.m_xcp_metrics); ++xcp)
                    {
                        for(std::size_t i = 0;
                            i < std::size(itr.m_xcp_metrics[xcp].vcn_busy); ++i)
                        {
                            counter_track::emplace(
                                _dev_id, addendum_blk(i, "VCN Activity", xcp), "%");
                        }
                    }
                }
            }
            if(_settings.jpeg_activity)
            {
                if(itr.m_xcp_metrics.empty())
                {
                    ROCPROFSYS_VERBOSE(
                        1, "No JPEG activity data collected from device %u\n", _dev_id);
                }
                else if(gpu::is_jpeg_activity_supported(_dev_id))
                {
                    for(std::size_t i = 0; i < std::size(itr.m_xcp_metrics[0].jpeg_busy);
                        ++i)
                        counter_track::emplace(_dev_id, addendum_blk(i, "JPEG Activity"),
                                               "%");
                }
                else
                {
                    for(std::size_t xcp = 0; xcp < std::size(itr.m_xcp_metrics); ++xcp)
                    {
                        for(std::size_t i = 0;
                            i < std::size(itr.m_xcp_metrics[xcp].jpeg_busy); ++i)
                            counter_track::emplace(
                                _dev_id, addendum_blk(i, "JPEG Activity", xcp), "%");
                    }
                }
            }
        };

        auto write_perfetto_metrics = [&]() {
            size_t track_index = 0;

            if(_settings.busy)
            {
                TRACE_COUNTER("device_busy_gfx",
                              counter_track::at(_dev_id, track_index++), _ts, _gfxbusy);
                TRACE_COUNTER("device_busy_umc",
                              counter_track::at(_dev_id, track_index++), _ts, _umcbusy);
                TRACE_COUNTER("device_busy_mm", counter_track::at(_dev_id, track_index++),
                              _ts, _mmbusy);
            }
            if(_settings.temp)
            {
                TRACE_COUNTER("device_temp", counter_track::at(_dev_id, track_index++),
                              _ts, _temp);
            }
            if(_settings.power)
            {
                TRACE_COUNTER("device_power", counter_track::at(_dev_id, track_index++),
                              _ts, _power);
            }
            if(_settings.mem_usage)
            {
                TRACE_COUNTER("device_memory_usage",
                              counter_track::at(_dev_id, track_index++), _ts, _usage);
            }

            if(_settings.vcn_activity && !itr.m_xcp_metrics.empty())
            {
                // Iterate over all XCPs and their VCN busy/activity values
                for(const auto& metrics : itr.m_xcp_metrics)
                {
                    for(const auto& vcn_val : metrics.vcn_busy)
                    {
                        TRACE_COUNTER("device_vcn_activity",
                                      counter_track::at(_dev_id, track_index++), _ts,
                                      vcn_val);
                    }
                }
            }

            if(_settings.jpeg_activity && !itr.m_xcp_metrics.empty())
            {
                // Iterate over all XCPs and their JPEG busy/activity values
                for(const auto& metrics : itr.m_xcp_metrics)
                {
                    for(const auto& jpeg_val : metrics.jpeg_busy)
                    {
                        TRACE_COUNTER("device_jpeg_activity",
                                      counter_track::at(_dev_id, track_index++), _ts,
                                      jpeg_val);
                    }
                }
            }
        };

        if(use_perfetto)
        {
            setup_perfetto_counter_tracks();
            write_perfetto_metrics();
        }

        if(use_rocpd)
        {
            rocpd_process_smi_pmc_events(_dev_id, _settings, _ts, _mmbusy, _temp, _power,
                                         _usage);
        }
    }
}

/**
 * @brief Initialize AMD SMI sampling and configure per-device metrics/tracking.
 *
 * Acquires the AMD SMI category mutex and, if AMD SMI is enabled and not already
 * initialized, attempts to initialize the AMD SMI library and enumerate devices.
 * Parses the sampling GPU specification and the ROCPROFSYS_AMD_SMI_METRICS setting
 * to build the set of devices to sample and which metrics to enable per device.
 * On success marks the AMD SMI subsystem as initialized and invokes data::setup().
 *
 * Behavior and side effects:
 * - Returns immediately if already initialized or AMD SMI use is disabled.
 * - Calls gpu::initialize_amdsmi() and logs/returns early if AMD SMI is unavailable.
 * - Sets data::device_count and data::device_list based on the sampling GPU spec.
 * - Interprets special strings "off"/"none", "on"/"all", numeric IDs, and ranges (N-M).
 * - Configures per-device metric enablement according to ROCPROFSYS_AMD_SMI_METRICS.
 * - Marks the subsystem initialized (is_initialized() set to true) and calls data::setup().
 * - Exceptions thrown during configuration are caught; on error device_list is cleared
 *   and initialization is not completed.
 */

void
setup()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(is_initialized() || !get_use_amd_smi()) return;

    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    if(!gpu::initialize_amdsmi())
    {
        ROCPROFSYS_WARNING_F(0,
                             "AMD SMI is not available. Disabling AMD SMI sampling...");
        return;
    }

    amdsmi_version_t _version = get_version();
    ROCPROFSYS_VERBOSE_F(0, "AMD SMI version: %u.%u.%u - str: %s.\n", _version.major,
                         _version.minor, _version.release, _version.build);

    data::device_count = gpu::device_count();

    auto _devices_v = get_sampling_gpus();
    for(auto& itr : _devices_v)
        itr = tolower(itr);
    if(_devices_v == "off")
        _devices_v = "none";
    else if(_devices_v == "on")
        _devices_v = "all";
    bool _all_devices = _devices_v.find("all") != std::string::npos || _devices_v.empty();
    bool _no_devices  = _devices_v.find("none") != std::string::npos;

    std::set<uint32_t> _devices = {};
    auto               _emplace = [&_devices](auto idx) {
        if(idx < data::device_count) _devices.emplace(idx);
    };

    if(_all_devices)
    {
        for(uint32_t i = 0; i < data::device_count; ++i)
            _emplace(i);
    }
    else if(!_no_devices)
    {
        auto _enabled = tim::delimit(_devices_v, ",; \t");
        for(auto&& itr : _enabled)
        {
            if(itr.find_first_not_of("0123456789-") != std::string::npos)
            {
                ROCPROFSYS_THROW("Invalid GPU specification: '%s'. Only numerical values "
                                 "(e.g., 0) or ranges (e.g., 0-7) are permitted.",
                                 itr.c_str());
            }

            if(itr.find('-') != std::string::npos)
            {
                auto _v = tim::delimit(itr, "-");
                ROCPROFSYS_CONDITIONAL_THROW(_v.size() != 2,
                                             "Invalid GPU range specification: '%s'. "
                                             "Required format N-M, e.g. 0-4",
                                             itr.c_str());
                for(auto i = std::stoul(_v.at(0)); i < std::stoul(_v.at(1)); ++i)
                    _emplace(i);
            }
            else
            {
                _emplace(std::stoul(itr));
            }
        }
    }

    data::device_list = _devices;

    auto _metrics = get_setting_value<std::string>("ROCPROFSYS_AMD_SMI_METRICS");

    try
    {
        for(auto itr : _devices)
        {
            // Enable selected metrics only
            if((_metrics && !_metrics->empty()) && (*_metrics != "all"))
            {
                using key_pair_t     = std::pair<std::string_view, bool&>;
                const auto supported = std::unordered_map<std::string_view, bool&>{
                    key_pair_t{ "busy", get_settings(itr).busy },
                    key_pair_t{ "temp", get_settings(itr).temp },
                    key_pair_t{ "power", get_settings(itr).power },
                    key_pair_t{ "mem_usage", get_settings(itr).mem_usage },
                    key_pair_t{ "vcn_activity", get_settings(itr).vcn_activity },
                    key_pair_t{ "jpeg_activity", get_settings(itr).jpeg_activity },
                };

                // Initialize all metrics to false
                for(auto& it : supported)
                    it.second = false;

                // Parse list of metrics enabled by the user
                if(*_metrics != "none")
                {
                    for(const auto& metric : tim::delimit(*_metrics, ",;:\t\n "))
                    {
                        auto iitr = supported.find(metric);
                        if(iitr == supported.end())
                            ROCPROFSYS_FAIL_F("unsupported amd-smi metric: %s\n",
                                              metric.c_str());
                        ROCPROFSYS_VERBOSE_F(
                            1, "Enabling amd-smi metric '%s' on device [%u]\n",
                            metric.c_str(), itr);
                        iitr->second = true;
                    }
                }
            }
        }

        is_initialized() = true;
        data::setup();
    } catch(std::runtime_error& _e)
    {
        ROCPROFSYS_VERBOSE(0, "Exception thrown when initializing amd-smi: %s\n",
                           _e.what());
        data::device_list = {};
    }
}

void
shutdown()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(!is_initialized()) return;
    ROCPROFSYS_VERBOSE_F(1, "Shutting down amd-smi...\n");

    try
    {
        if(data::shutdown())
        {
            ROCPROFSYS_AMD_SMI_CALL(amdsmi_shut_down());
        }
    } catch(std::runtime_error& _e)
    {
        ROCPROFSYS_VERBOSE(0, "Exception thrown when shutting down amd-smi: %s\n",
                           _e.what());
    }

    is_initialized() = false;
}

/**
 * @brief Post-process collected AMD SMI samples for all enabled devices.
 *
 * Iterates the configured device_list and invokes data::post_process(device_id)
 * for each device.
 */
void
post_process()
{
    for(auto itr : data::device_list)
    {
        ROCPROFSYS_VERBOSE(2, "Post-processing amd-smi data for device: %d", itr);
        data::post_process(itr);
    }
}

/**
 * @brief Get the number of available GPU devices.
 *
 * @return uint32_t The count of GPU devices as reported by gpu::device_count().
 */
uint32_t
device_count()
{
    return gpu::device_count();
}
}  // namespace amd_smi
}  // namespace rocprofsys

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_gfx>),
    true, double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_umc>),
    true, double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_mm>),
    true, double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_temp>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_power>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_memory>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_vcn>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_jpeg>), true,
    double)
