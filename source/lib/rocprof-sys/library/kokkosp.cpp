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

#include <optional>
#define TIMEMORY_KOKKOSP_POSTFIX ROCPROFSYS_PUBLIC_API

#include "api.hpp"
#include "core/agent_manager.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/node_info.hpp"
#include "core/perfetto.hpp"
#include "core/rocpd/data_processor.hpp"
#include "core/rocpd/json.hpp"
#include "library/components/category_region.hpp"
#include "library/runtime.hpp"

#include <timemory/api/kokkosp.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/utility/procfs/maps.hpp>
#include <timemory/utility/types.hpp>

#include <cstdlib>
#include <sstream>
#include <string>

namespace kokkosp  = ::tim::kokkosp;
namespace category = ::tim::category;
namespace comp     = ::rocprofsys::component;

using kokkosp_region = comp::local_category_region<category::kokkos>;

//--------------------------------------------------------------------------------------//

namespace tim
{
template <>
inline auto
invoke_preinit<kokkosp::memory_tracker>(long)
{
    kokkosp::memory_tracker::label()       = "kokkos_memory";
    kokkosp::memory_tracker::description() = "Kokkos Memory tracker";
}
}  // namespace tim

//--------------------------------------------------------------------------------------//

namespace
{
std::string kokkos_banner =
    "#---------------------------------------------------------------------------#";

//--------------------------------------------------------------------------------------//

inline void
setup_kernel_logger()
{
    if((tim::settings::debug() && tim::settings::verbose() >= 3) ||
       rocprofsys::config::get_use_kokkosp_kernel_logger())
    {
        kokkosp::logger_t::get_initializer() = [](kokkosp::logger_t& _obj) {
            _obj.initialize<kokkosp::kernel_logger>();
        };
    }
}

}  // namespace

namespace
{
bool                     _standalone_initialized = false;
bool                     _kp_deep_copy           = false;
size_t                   _name_len_limit         = 0;
std::string              _kp_prefix              = {};
std::vector<std::string> _initialize_arguments   = {};

template <typename Tp>
void
set_invalid_id(Tp* _v)
{
    constexpr bool is32 = std::is_same<Tp, uint32_t>::value;
    constexpr bool is64 = std::is_same<Tp, uint64_t>::value;
    static_assert(is32 || is64, "only support uint32_t or uint64_t");

    *_v = std::numeric_limits<Tp>::max();
}

template <typename Tp>
bool
is_invalid_id(Tp _v)
{
    constexpr bool is32 = std::is_same<Tp, uint32_t>::value;
    constexpr bool is64 = std::is_same<Tp, uint64_t>::value;
    static_assert(is32 || is64, "only support uint32_t or uint64_t");

    return (_v == std::numeric_limits<Tp>::max());
}

template <typename Tp>
auto
strlength(Tp&& _v)
{
    using type = ::tim::concepts::unqualified_type_t<Tp>;
    if constexpr(std::is_same<type, std::string_view>::value ||
                 std::is_same<type, std::string>::value)
        return _v.length();
    else
        return strnlen(_v, std::max<size_t>(_name_len_limit, 1));
}

template <typename Arg, typename... Args>
/**
 * @brief Determine whether one or more profiling name components violate naming rules.
 *
 * Evaluates the provided name arguments and returns true when they should be rejected for
 * profiling:
 * - If causal profiling is enabled and the first argument begins with "Kokkos::" or any
 *   argument contains "Space::", the name is considered a violation.
 * - If the combined length of all name arguments is zero, the name is considered a violation.
 * - If the global name-length limit (_name_len_limit) is zero, names are always allowed.
 * - Otherwise, the name is a violation when the total length is greater than or equal to
 *   _name_len_limit.
 *
 * The function does not modify its inputs.
 *
 * @tparam Arg, Args Types convertible to strings (e.g., string, string_view, C-string).
 * @param _arg First name component.
 * @param _args Remaining name components.
 * @return true if the supplied name components violate the naming rules; false otherwise.
 */
bool
violates_name_rules(Arg&& _arg, Args&&... _args)
{
    // for causal profiling we only consider callbacks which are explicitly named
    if(rocprofsys::config::get_use_causal() &&
       (std::string_view{ _arg }.find("Kokkos::") == 0 ||
        std::string_view{ _arg }.find("Space::") != std::string_view::npos))
        return true;

    size_t _len =
        (strlength(std::forward<Arg>(_arg)) + ... + strlength(std::forward<Args>(_args)));

    // ignore labels without names
    if(_len == 0)
        return true;
    else if(_name_len_limit == 0)
        return false;

    return (_len >= _name_len_limit);
}
}  // namespace

namespace
{
/**
 * @brief Get the singleton rocpd data processor.
 *
 * Returns the global rocprofsys::rocpd::data_processor instance used to submit
 * Kokkos events and samples to the rocpd pipeline.
 *
 * @return rocprofsys::rocpd::data_processor& Reference to the singleton data processor.
 */
rocprofsys::rocpd::data_processor&
get_data_processor()
{
    return rocprofsys::rocpd::data_processor::get_instance();
}

/**
 * @brief Register the Kokkos category with the rocpd data processor.
 *
 * Inserts a category entry into the global rocpd data_processor using the
 * predefined category id and name for Kokkos. This ensures subsequent Kokkos
 * events and samples are recorded under the correct category.
 */
void
rocpd_initialize_kokkos_category()
{
    get_data_processor().insert_category(
        rocprofsys::category_enum_id<category::kokkos>::value,
        rocprofsys::trait::name<category::kokkos>::value);
}

/**
 * @brief Register a Kokkos track with the global rocpd data processor.
 *
 * Inserts a tracking entry for the Kokkos category into the shared
 * rocpd::data_processor using the current node id and process id. The
 * track is created without an associated thread id (thread id left null).
 *
 * This makes Kokkos events and samples identifiable in the rocpd data stream
 * under the Kokkos category for the current node and process.
 */
void
rocpd_initialize_kokkos_track()
{
    auto& data_processor = get_data_processor();
    auto& n_info         = rocprofsys::node_info::get_instance();
    auto  thread_id      = std::nullopt;

    data_processor.insert_track(rocprofsys::trait::name<category::kokkos>::value,
                                n_info.id, getpid(), thread_id);
}

/**
 * @brief Record a Kokkos-related event into the rocpd data processor.
 *
 * Creates a small JSON metadata object containing the event name, type, and
 * target, inserts it as an event in the rocpd data processor under the
 * Kokkos category, and then inserts a timestamped sample that references the
 * newly created event.
 *
 * @param name Human-readable name of the Kokkos event (e.g., kernel or region name).
 * @param event_type Short tag describing the event type (e.g., "sync", "modify").
 * @param target Target device or domain for the event (e.g., "device", "host").
 * @param timestamp_ns Monotonic timestamp for the sample, in nanoseconds.
 */
void
rocpd_process_kokkos_event(const char* name, const char* event_type, const char* target,
                           uint64_t timestamp_ns)
{
    auto& data_processor = get_data_processor();
    auto  event_metadata = rocpd::json::create();

    event_metadata->set("name", name);
    event_metadata->set("event_type", event_type);
    event_metadata->set("target", target);

    auto event_id = data_processor.insert_event(
        rocprofsys::category_enum_id<category::kokkos>::value, 0, 0, 0, "{}", "{}",
        event_metadata->to_string().c_str());
    data_processor.insert_sample(rocprofsys::trait::name<category::kokkos>::value,
                                 timestamp_ns, event_id, "{}");
}

}  // namespace
//--------------------------------------------------------------------------------------//

extern "C"
{
    struct Kokkos_Tools_ToolSettings
    {
        bool requires_global_fencing;
        bool padding[255];
    };

    void kokkosp_request_tool_settings(const uint32_t,
                                       Kokkos_Tools_ToolSettings*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_dual_view_sync(const char*, const void* const,
                                bool) ROCPROFSYS_PUBLIC_API;
    void kokkosp_dual_view_modify(const char*, const void* const,
                                  bool) ROCPROFSYS_PUBLIC_API;

    void kokkosp_print_help(char*) {}

    void kokkosp_parse_args(int argc, char** argv)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(!rocprofsys::config::settings_are_configured() &&
           rocprofsys::get_state() < rocprofsys::State::Active)
        {
            _standalone_initialized = true;

            ROCPROFSYS_BASIC_VERBOSE_F(0, "Parsing arguments...\n");
            std::string _command_line = {};
            for(int i = 0; i < argc; ++i)
            {
                _initialize_arguments.emplace_back(argv[i]);
                _command_line.append(" ").append(argv[i]);
            }
            if(_command_line.length() > 1) _command_line = _command_line.substr(1);
            tim::set_env("ROCPROFSYS_COMMAND_LINE", _command_line, 0);
        }
    }

    void kokkosp_declare_metadata(const char* key, const char* value)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        tim::manager::add_metadata(key, value);
    }

    void kokkosp_request_tool_settings(const uint32_t             _version,
                                       Kokkos_Tools_ToolSettings* _settings)
    {
        if(_version > 0) _settings->requires_global_fencing = false;
    }

    /**
     * @brief Initialize the rocprof-sys Kokkos connector.
     *
     * Initializes connector state when the Kokkos Tools library is loaded. Depending
     * runtime configuration and process state this may perform a standalone
     * initialization of rocprof-sys (including hidden-mode init and trace push),
     * register rocpd Kokkos category/track, enable the internal memory tracker, set
     * up kernel logging, and read connector-specific configuration values
     * (name length limit, profiling prefix, deep-copy tracking).
     *
     * This function has global side effects: it may modify connector global state,
     * initialize rocprof-sys, and install rocpd metadata/track entries when enabled.
     *
     * @param loadSeq Loader sequence number passed by Kokkos.
     * @param interfaceVer Kokkos tools interface version provided by the runtime.
     * @param devInfoCount Count of device info entries (consumed but not used).
     * @param deviceInfo Pointer to device info (consumed but not used).
     *
     * @note The function will abort the process if it detects an incompatible or
     * duplicate rocprof-sys library already loaded (to prevent duplicate
     * collection).
     */
    void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                              const uint32_t devInfoCount, void* deviceInfo)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        tim::consume_parameters(devInfoCount, deviceInfo);

        ROCPROFSYS_BASIC_VERBOSE_F(
            0,
            "Initializing rocprof-sys kokkos connector (sequence %d, version: %llu)... ",
            loadSeq, (unsigned long long) interfaceVer);

        if(_standalone_initialized ||
           (!rocprofsys::config::settings_are_configured() &&
            rocprofsys::get_state() < rocprofsys::State::Active))
        {
            auto _kokkos_profile_lib = tim::get_env<std::string>("KOKKOS_TOOLS_LIBS");
            if(_kokkos_profile_lib.find("librocprof-sys.so") != std::string::npos)
            {
                auto _maps = tim::procfs::read_maps(tim::process::get_id());
                auto _libs = std::set<std::string>{};
                for(auto& itr : _maps)
                {
                    auto&& _path = itr.pathname;
                    if(!_path.empty() && _path.at(0) != '[' &&
                       rocprofsys::filepath::exists(_path))
                        _libs.emplace(_path);
                }
                for(const auto& itr : _libs)
                {
                    if(itr.find("librocprof-sys-dl.so") != std::string::npos)
                    {
                        std::stringstream _libs_str{};
                        for(const auto& litr : _libs)
                            _libs_str << "    " << litr << "\n";
                        ROCPROFSYS_ABORT(
                            "%s was invoked with librocprof-sys.so as the "
                            "KOKKOS_TOOLS_LIBS.\n"
                            "However, librocprof-sys-dl.so has already been loaded by "
                            "the process.\nTo avoid duplicate collections culminating "
                            "is an error, please set KOKKOS_TOOLS_LIBS=%s.\nLoaded "
                            "libraries:\n%s",
                            __FUNCTION__, itr.c_str(), _libs_str.str().c_str());
                    }
                }
            }

            ROCPROFSYS_BASIC_VERBOSE_F(0, "Initializing rocprof-sys (standalone)... ");
            auto _mode = tim::get_env<std::string>("ROCPROFSYS_MODE", "trace");
            auto _arg0 = (_initialize_arguments.empty()) ? std::string{ "unknown" }
                                                         : _initialize_arguments.at(0);

            _standalone_initialized = true;
            rocprofsys_set_mpi_hidden(false, false);
            rocprofsys_init_hidden(_mode.c_str(), false, _arg0.c_str());
            rocprofsys_push_trace_hidden("kokkos_main");

            if(rocprofsys::get_use_rocpd())
            {
                rocpd_initialize_kokkos_category();
                rocpd_initialize_kokkos_track();
            }
        }

        setup_kernel_logger();

        tim::trait::runtime_enabled<kokkosp::memory_tracker>::set(
            rocprofsys::config::get_use_timemory());

        if(rocprofsys::get_verbose() >= 0)
        {
            fprintf(stderr, "%sDone\n%s", tim::log::color::info(),
                    tim::log::color::end());
        }

        _name_len_limit = rocprofsys::config::get_setting_value<int64_t>(
                              "ROCPROFSYS_KOKKOSP_NAME_LENGTH_MAX")
                              .value_or(_name_len_limit);
        _kp_prefix = rocprofsys::config::get_setting_value<std::string>(
                         "ROCPROFSYS_KOKKOSP_PREFIX")
                         .value_or(_kp_prefix);

        _kp_deep_copy =
            rocprofsys::config::get_setting_value<bool>("ROCPROFSYS_KOKKOSP_DEEP_COPY")
                .value_or(_kp_deep_copy);
    }

    void kokkosp_finalize_library()
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(_standalone_initialized)
        {
            rocprofsys_pop_trace_hidden("kokkos_main");
            ROCPROFSYS_VERBOSE_F(
                0, "Finalizing kokkos rocprof-sys connector (standalone)...\n");
            rocprofsys_finalize_hidden();
        }
        else
        {
            ROCPROFSYS_VERBOSE_F(0, "Finalizing kokkos rocprof-sys connector... ");
            kokkosp::cleanup();
            if(rocprofsys::get_verbose() >= 0) fprintf(stderr, "Done\n");
        }
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid)
    {
        if(violates_name_rules(name)) return set_invalid_id(kernid);

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname = (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                         ? JOIN(" ", _kp_prefix, name, "[for]")
                         : JOIN(" ", _kp_prefix, name, JOIN("", "[for][dev", devid, ']'));
        *kernid    = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_parallel_for(uint64_t kernid)
    {
        if(is_invalid_id(kernid)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid)
    {
        if(violates_name_rules(name)) return set_invalid_id(kernid);

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? JOIN(" ", _kp_prefix, name, "[reduce]")
                : JOIN(" ", _kp_prefix, name, JOIN("", "[reduce][dev", devid, ']'));
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_parallel_reduce(uint64_t kernid)
    {
        if(is_invalid_id(kernid)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid)
    {
        if(violates_name_rules(name)) return set_invalid_id(kernid);

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? JOIN(" ", _kp_prefix, name, "[scan]")
                : JOIN(" ", _kp_prefix, name, JOIN("", "[scan][dev", devid, ']'));
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_parallel_scan(uint64_t kernid)
    {
        if(is_invalid_id(kernid)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_fence(const char* name, uint32_t devid, uint64_t* kernid)
    {
        if(violates_name_rules(name)) return set_invalid_id(kernid);

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? JOIN(" ", _kp_prefix, name, "[fence]")
                : JOIN(" ", _kp_prefix, name, JOIN("", "[fence][dev", devid, ']'));
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_fence(uint64_t kernid)
    {
        if(is_invalid_id(kernid)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_push_profile_region(const char* name)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name);
        kokkosp::get_profiler_stack<kokkosp_region>()
            .emplace_back(kokkosp::profiler_t<kokkosp_region>(name))
            .start();
    }

    void kokkosp_pop_profile_region()
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        if(kokkosp::get_profiler_stack<kokkosp_region>().empty()) return;
        kokkosp::get_profiler_stack<kokkosp_region>().back().stop();
        kokkosp::get_profiler_stack<kokkosp_region>().pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_create_profile_section(const char* name, uint32_t* secid)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        *secid     = kokkosp::get_unique_id();
        auto pname = std::string{ name };
        kokkosp::create_profiler<kokkosp_region>(name, *secid);
    }

    void kokkosp_destroy_profile_section(uint32_t secid)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::destroy_profiler<kokkosp_region>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_start_profile_section(uint32_t secid)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(1, __FUNCTION__, secid);
        kokkosp::start_profiler<kokkosp_region>(secid);
    }

    void kokkosp_stop_profile_section(uint32_t secid)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, secid);
        kokkosp::stop_profiler<kokkosp_region>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_allocate_data(const SpaceHandle space, const char* label,
                               const void* const ptr, const uint64_t size)
    {
        if(violates_name_rules(label)) return;
        if(rocprofsys::config::get_use_causal()) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(0, __FUNCTION__, space.name, label,
                                 JOIN("", '[', ptr, ']'), size);
        auto pname =
            JOIN(" ", _kp_prefix, label, JOIN("", '[', space.name, "][allocate]"));
        kokkosp::profiler_alloc_t<>{ pname }.store(std::plus<int64_t>{}, size);
        kokkosp::profiler_t<kokkosp_region>{ pname }.mark();
    }

    void kokkosp_deallocate_data(const SpaceHandle space, const char* label,
                                 const void* const ptr, const uint64_t size)
    {
        if(violates_name_rules(label)) return;
        if(rocprofsys::config::get_use_causal()) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(0, __FUNCTION__, space.name, label,
                                 JOIN("", '[', ptr, ']'), size);
        auto pname =
            JOIN(" ", _kp_prefix, label, JOIN("", '[', space.name, "][deallocate]"));
        kokkosp::profiler_alloc_t<>{ pname }.store(std::plus<int64_t>{}, size);
        kokkosp::profiler_t<kokkosp_region>{ pname }.mark();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_deep_copy(SpaceHandle dst_handle, const char* dst_name,
                                 const void* dst_ptr, SpaceHandle src_handle,
                                 const char* src_name, const void* src_ptr, uint64_t size)
    {
        if(!_kp_deep_copy || rocprofsys::config::get_use_causal()) return;
        if(violates_name_rules(dst_name, src_name)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(1, __FUNCTION__, dst_handle.name, dst_name,
                                 JOIN("", '[', dst_ptr, ']'), src_handle.name, src_name,
                                 JOIN("", '[', src_ptr, ']'), size);

        auto name = JOIN(" ", _kp_prefix, JOIN('=', dst_handle.name, dst_name), "<-",
                         JOIN('=', src_handle.name, src_name), "[deep_copy]");

        auto& _data = kokkosp::get_profiler_stack<kokkosp_region>();
        _data.emplace_back(name);
        _data.back().audit(dst_handle, dst_name, dst_ptr, src_handle, src_name, src_ptr,
                           size);
        _data.back().start();
        _data.back().store(tim::mpl::piecewise_select<kokkosp::memory_tracker>{},
                           std::plus<int64_t>{}, size);
    }

    void kokkosp_end_deep_copy()
    {
        if(!_kp_deep_copy || rocprofsys::config::get_use_causal()) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        auto& _data = kokkosp::get_profiler_stack<kokkosp_region>();
        if(_data.empty()) return;
        _data.back().store(tim::mpl::piecewise_select<kokkosp::memory_tracker>{},
                           std::minus<int64_t>{}, 0);
        _data.back().stop();
        _data.pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_profile_event(const char* name)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto _name = tim::get_hash_identifier_fast(tim::add_hash_id(name));
        kokkosp::profiler_t<kokkosp_region>{ _name }.mark();
    }

    /**
     * @brief Record a Kokkos dual-view synchronization event for tracing and profiling.
     *
     * Captures a high-resolution timestamp and, depending on runtime configuration,
     * emits an instant Perfetto trace event, records a causal/profile mark, and/or
     * forwards a Kokkos event to the rocpd data processor. If the provided name
     * violates configured naming rules, the call is a no-op.
     *
     * @param label Human-readable region/label for the dual-view sync (used in traces and events).
     * @param /*unused*/ Unused pointer parameter retained for API compatibility.
     * @param is_device If true, the event is marked as targeting the device; otherwise targets the host.
     */

    void kokkosp_dual_view_sync(const char* label, const void* const, bool is_device)
    {
        if(violates_name_rules(label)) return;

        auto timestamp = tim::get_clock_real_now<uint64_t, std::nano>();

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(rocprofsys::config::get_use_perfetto())
        {
            auto _name = tim::get_hash_identifier_fast(
                tim::add_hash_id(JOIN(" ", _kp_prefix, label, "[dual_view_sync]")));
            TRACE_EVENT_INSTANT("user", ::perfetto::StaticString{ _name.data() },
                                "target", (is_device) ? "device" : "host");
        }
        else if(rocprofsys::config::get_use_causal())
        {
            auto _name = tim::get_hash_identifier_fast(tim::add_hash_id(JOIN(
                "", label, " [dual_view_sync][", (is_device) ? "device" : "host", "]")));
            kokkosp::profiler_t<kokkosp_region>{ _name }.mark();
        }

        if(rocprofsys::config::get_use_rocpd())
        {
            rocpd_process_kokkos_event(JOIN(" ", _kp_prefix, label).c_str(),
                                       "[dual_view_sync]",
                                       (is_device) ? "device" : "host", timestamp);
        }
    }

    /**
     * @brief Record a "dual view modify" event for a Kokkos DualView.
     *
     * Records an instantaneous "dual_view_modify" event targeted at either the device
     * or the host, depending on runtime configuration. Depending on which backends
     * are enabled, this will emit a Perfetto instant trace, mark a causal profiler
     * entry, and/or submit a rocpd event with a high-resolution timestamp.
     *
     * If the provided label violates configured naming rules, the function is a no-op.
     *
     * @param label Human-readable label identifying the DualView.
     * @param is_device True if the modification target is the device; false for host.
     */
    void kokkosp_dual_view_modify(const char* label, const void* const, bool is_device)
    {
        if(violates_name_rules(label)) return;

        auto timestamp = tim::get_clock_real_now<uint64_t, std::nano>();
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(rocprofsys::config::get_use_perfetto())
        {
            auto _name = tim::get_hash_identifier_fast(
                tim::add_hash_id(JOIN(" ", _kp_prefix, label, "[dual_view_modify]")));
            TRACE_EVENT_INSTANT("user", ::perfetto::StaticString{ _name.data() },
                                "target", (is_device) ? "device" : "host");
        }
        else if(rocprofsys::config::get_use_causal())
        {
            auto _name = tim::get_hash_identifier_fast(
                tim::add_hash_id(JOIN(" ", _kp_prefix, label, "[dual_view_modify][",
                                      (is_device) ? "device" : "host", "]")));
            kokkosp::profiler_t<kokkosp_region>{ _name }.mark();
        }

        if(rocprofsys::config::get_use_rocpd())
        {
            rocpd_process_kokkos_event(JOIN(" ", _kp_prefix, label).c_str(),
                                       "[dual_view_modify]",
                                       (is_device) ? "device" : "host", timestamp);
        }
    }

    //----------------------------------------------------------------------------------//
}

TIMEMORY_INITIALIZE_STORAGE(kokkosp::memory_tracker)
