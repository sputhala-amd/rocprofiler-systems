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

#pragma once

#include "core/common.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"

#include <cstdint>
#include <map>
#include <string>

namespace rocprofsys
{
namespace component
{
// this is used to wrap MPI_Init and MPI_Init_thread
struct mpi_gotcha : comp::base<mpi_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 5;
    using hash_array_t                      = std::array<size_t, gotcha_capacity>;
    using comm_t        = tim::mpi::comm_t;
    using gotcha_data_t = comp::gotcha_data;

    ROCPROFSYS_DEFAULT_OBJECT(mpi_gotcha)

    explicit mpi_gotcha(const gotcha_data_t&);

    // string id for component
    // static std::string label() { return "mpi_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    // operator overloads for MPI functions
    int operator()(int (*)(int*, char***), int*, char***) const;
    int operator()(int (*)(int*, char***, int, int*), int*, char***, int, int*) const;
    int operator()(int (*)(void)) const;
    int operator()(int (*)(comm_t, int*), comm_t, int*) const;

    // without these you will get a verbosity level 1 warning
    // static void start() {}
    // static void stop() {}

    static bool      update();
    static uintptr_t null_comm() { return std::numeric_limits<uintptr_t>::max(); }
    static void      disable_comm_intercept();

private:
    mutable int       m_rank     = 0;
    mutable int       m_size     = 1;
    mutable int*      m_rank_ptr = nullptr;
    mutable int*      m_size_ptr = nullptr;
    mutable uintptr_t m_comm_val = null_comm();
    
    static bool          is_disabled();
    static hash_array_t& get_hashes();

    template <typename... Args>
    auto operator()(uintptr_t&&, int (*)(Args...), Args...) const;

    mutable bool         m_protect = false;
    const gotcha_data_t* m_data    = nullptr;

};
}  // namespace component

using mpi_gotcha_t = comp::gotcha<5, tim::component_tuple<component::mpi_gotcha>, project::rocprofsys>;
}  // namespace rocprofsys
