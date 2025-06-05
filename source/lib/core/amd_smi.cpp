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

#include "core/amd_smi.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/gpu.hpp"
#include "timemory.hpp"
#include <cassert>
#include <chrono>
#include <ios>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>

#define ROCPROFSYS_AMD_SMI_CALL(...)                                                     \
    ::rocprofsys::amd_smi::check_error(__FILE__, __LINE__, __VA_ARGS__)

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0
namespace rocprofsys
{
namespace amd_smi
{
namespace
{
auto&
get_settings(uint32_t _dev_id)
{
    static auto _v = std::unordered_map<uint32_t, amd_smi_settings>{};
    return _v[_dev_id];
}

std::string
get_setting_name(std::string _v)
{
    constexpr auto _prefix = tim::string_view_t{ "rocprofsys_" };
    for(auto& itr : _v)
        itr = tolower(itr);
    auto _pos = _v.find(_prefix);
    if(_pos == 0) return _v.substr(_prefix.length());
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

#    define ROCPROFSYS_CONFIG_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE, ...)   \
        [&]() {                                                                          \
            auto _ret = _config->insert<TYPE, TYPE>(                                     \
                ENV_NAME, get_setting_name(ENV_NAME), DESCRIPTION,                       \
                TYPE{ INITIAL_VALUE },                                                   \
                std::set<std::string>{ "custom", "rocprofsys", "librocprof-sys",         \
                                       __VA_ARGS__ });                                   \
            if(!_ret.second)                                                             \
            {                                                                            \
                ROCPROFSYS_PRINT("Warning! Duplicate setting: %s / %s\n",                \
                                 get_setting_name(ENV_NAME).c_str(), ENV_NAME);          \
            }                                                                            \
            return _config->find(ENV_NAME)->second;                                      \
        }()
}  // namespace

std::set<uint32_t> amd_smi_config_data::device_list         = {};
size_t             amd_smi_config_data::device_count        = 0;
bool               amd_smi_config_data::track_jpeg_activity = false;
bool               amd_smi_config_data::track_vcn_activity  = false;

void
setup_config_check()
{
    if(!get_use_amd_smi() || !gpu::initialize_amdsmi())
    {
        ROCPROFSYS_WARNING_F(
            0, "AMD SMI is not available. Disabling jpeg and vcn activity tracking.");
        return;
    }
    // Get processors and handles
    amd_smi_config_data::device_count = gpu::get_processor_count();
    auto _devices_v                   = get_sampling_gpus();
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
        if(idx < amd_smi_config_data::device_count) _devices.emplace(idx);
    };

    // Add devices to be sampled
    if(_all_devices)
    {
        for(uint32_t i = 0; i < amd_smi_config_data::device_count; ++i)
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
    else if(_no_devices)
    {
        ROCPROFSYS_WARNING_F(0, "No AMD SMI devices specified. Disabling AMD SMI "
                                "sampling for all devices...");
        return;
    }
    // Set Device list
    amd_smi_config_data::device_list = _devices;
}

void
config_settings(const std::shared_ptr<settings>& _config)
{
    setup_config_check();
    std::string desc_metrics = "busy, temp, power, mem_usage";

#    define ROCPROFSYS_AMDSMI_GET(OPTION, FUNCTION, ...)                                 \
        if(OPTION)                                                                       \
        {                                                                                \
            try                                                                          \
            {                                                                            \
                ROCPROFSYS_AMD_SMI_CALL(FUNCTION(__VA_ARGS__), &OPTION);                 \
            } catch(std::runtime_error & _e)                                             \
            {                                                                            \
                ROCPROFSYS_VERBOSE_F(                                                    \
                    0, "[%s] Exception: %s. Disabling future samples from amd-smi...\n", \
                    #FUNCTION, _e.what());                                               \
            }                                                                            \
        }

    if(!amd_smi_config_data::device_list.empty())
    {
        for(const auto& dev_id : amd_smi_config_data::device_list)
        {
            amdsmi_gpu_metrics_t    _gpu_metrics{};
            amdsmi_processor_handle sample_handle = gpu::get_handle_from_id(dev_id);

            // Verify that metrics are available and set respective flag
            ROCPROFSYS_AMDSMI_GET(get_settings(dev_id).vcn_activity,
                                  amdsmi_get_gpu_metrics_info, sample_handle,
                                  &_gpu_metrics);
            ROCPROFSYS_AMDSMI_GET(get_settings(dev_id).jpeg_activity,
                                  amdsmi_get_gpu_metrics_info, sample_handle,
                                  &_gpu_metrics);

            if(_gpu_metrics.vcn_activity[0] != UINT16_MAX)
                amd_smi_config_data::track_vcn_activity = true;
            if(_gpu_metrics.jpeg_activity[0] != UINT16_MAX)
                amd_smi_config_data::track_jpeg_activity = true;
        }
#    undef ROCPROFSYS_AMDSMI_GET
    }

    if(amd_smi_config_data::track_jpeg_activity) desc_metrics += ", jpeg_activity";
    if(amd_smi_config_data::track_vcn_activity) desc_metrics += ", vcn_activity";

    desc_metrics += ". ";
    ROCPROFSYS_CONFIG_SETTING(
        std::string, "ROCPROFSYS_AMD_SMI_METRICS",
        "amd-smi metrics to collect: " + desc_metrics +
            "An empty value implies 'all' and 'none' suppresses all.",
        "busy, temp, power, mem_usage", "backend", "amd_smi", "rocm", "process_sampling");
}
}  // namespace amd_smi
}  // namespace rocprofsys

#else
namespace rocprofsys
{
namespace amd_smi
{
void
config_settings(const std::shared_ptr<settings>&)
{}
}  // namespace amd_smi
}  // namespace rocprofsys
#endif
