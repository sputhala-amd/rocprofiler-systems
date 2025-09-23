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

#include "node_info.hpp"
#include "debug.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sys/utsname.h>

namespace rocprofsys
{

/**
 * @brief Construct a node_info by reading machine and OS identity information.
 *
 * Attempts to initialize node identity from /etc/machine-id and the system
 * uname(2) information. On construction it:
 * - Tries to read the machine ID from /etc/machine-id; if the file cannot be
 *   opened a warning is emitted and initialization stops early.
 * - If the file is opened but no machine ID can be read, a warning is emitted
 *   and initialization continues with an empty machine_id.
 * - Computes `hash` as std::hash<std::string>(machine_id) normalized to
 *   fit in an int64_t, and `id` as `hash` modulo size_t max.
 * - Calls uname(2) to populate system_name, node_name, release, version,
 *   machine, and domain_name; if uname fails a warning is emitted and
 *   remaining fields are not populated.
 *
 * All warnings are emitted via ROCPROFSYS_WARNING. The constructor does not
 * throw exceptions.
 */
node_info::node_info()
{
    auto ifs = std::ifstream{ "/etc/machine-id" };
    if(!ifs.is_open())
    {
        ROCPROFSYS_WARNING(0, "Error: Unable to open /etc/machine-id!");
        return;
    }
    if(!(ifs >> machine_id) || machine_id.empty())
    {
        ROCPROFSYS_WARNING(0, "Error: Unable to read machine ID from /etc/machine-id!");
    }

    hash = std::hash<std::string>{}(machine_id) % std::numeric_limits<int64_t>::max();
    id   = hash % std::numeric_limits<size_t>::max();

    struct utsname _sys_info;
    if(uname(&_sys_info))
    {
        ROCPROFSYS_WARNING(0, "Error: Unable to get system information!");
        return;
    }

    system_name = _sys_info.sysname;
    node_name   = _sys_info.nodename;
    release     = _sys_info.release;
    version     = _sys_info.version;
    machine     = _sys_info.machine;
    domain_name = _sys_info.domainname;
}

/**
 * @brief Returns the program-wide singleton node_info instance.
 *
 * The instance is created on first call as a function-local static and then
 * reused for the lifetime of the program. Use this to obtain a reference to
 * the shared node_info containing the system/machine identity information.
 *
 * @return node_info& Reference to the single shared node_info instance.
 */
node_info&
node_info::get_instance()
{
    static node_info instance;
    return instance;
}

}  // namespace rocprofsys
