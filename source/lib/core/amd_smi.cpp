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

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0
namespace rocprofsys
{
namespace amd_smi
{
namespace
{
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

// See hpp for more information
std::unordered_set<uint32_t> amd_smi_config_data::gpuID_vcn_support  = {};
std::unordered_set<uint32_t> amd_smi_config_data::gpuID_jpeg_support = {};

bool
setup_config_check()
{
    if(!get_use_amd_smi() || !gpu::initialize_amdsmi()) return false;

    // Get processors and activity support
    size_t device_count = gpu::get_processor_count();
    for(size_t i = 0; i < device_count; i++)
    {
        if(gpu::is_jpeg_activity_supported(i) || gpu::is_jpeg_busy_supported(i))
            amd_smi_config_data::gpuID_jpeg_support.insert(i);
        if(gpu::is_vcn_activity_supported(i) || gpu::is_vcn_busy_supported(i))
            amd_smi_config_data::gpuID_vcn_support.insert(i);
    }
    return true;
}

void
config_settings(const std::shared_ptr<settings>& _config)
{
    if(!setup_config_check()) return;

    std::string default_metrics = "busy, temp, power, mem_usage";
    // List of gpu's that support JPEG and VCN activity
    // As of now, no distinction between busy and activity shown in description
    std::string jpeg_support = "";
    std::string vcn_support  = "";

    if(!amd_smi_config_data::gpuID_jpeg_support.empty())
    {
        jpeg_support += ", jpeg_activity (GPUs:";
        for(const auto& id : amd_smi_config_data::gpuID_jpeg_support)
            jpeg_support += " " + std::to_string(id) + ",";
        jpeg_support.pop_back();
        jpeg_support += ")";
    }

    if(!amd_smi_config_data::gpuID_vcn_support.empty())
    {
        vcn_support += ", vcn_activity (GPUs:";
        for(const auto& id : amd_smi_config_data::gpuID_vcn_support)
            vcn_support += " " + std::to_string(id) + ",";
        vcn_support.pop_back();
        vcn_support += ")";
    }

    ROCPROFSYS_CONFIG_SETTING(
        std::string, "ROCPROFSYS_AMD_SMI_METRICS",
        "amd-smi metrics to collect: " + default_metrics + jpeg_support + vcn_support +
            ". " + "An empty value implies 'all' and 'none' suppresses all.",
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
