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

#include "core/timemory.hpp"
#include <chrono>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <ratio>
#include <thread>
#include <tuple>
#include <type_traits>


#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace amd_smi
{
void
config_settings(const std::shared_ptr<settings>&);

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0

bool
setup_config_check();

struct amd_smi_config_data
{
    static std::set<uint32_t> gpuID_vcn_activity_support;
    // static std::set<uint32_t> gpuID_vcn_busy_support;
    static std::set<uint32_t> gpuID_jpeg_activity_support;
    // static std::set<uint32_t> gpuID_jpeg_busy_support;

private:
    friend bool rocprofsys::amd_smi::setup_config_check();
    friend void rocprofsys::amd_smi::config_settings(const std::shared_ptr<settings>&);
};

#endif
}  // namespace amd_smi
}  // namespace rocprofsys
