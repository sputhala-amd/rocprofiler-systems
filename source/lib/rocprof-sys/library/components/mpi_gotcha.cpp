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

#include "library/components/mpi_gotcha.hpp"
#include "api.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/mproc.hpp"
#include "library/components/category_region.hpp"
#include "library/components/comm_data.hpp"

#include <timemory/backends/mpi.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/signals/signal_mask.hpp>
#include <timemory/utility/locking.hpp>

#include <cstdint>
#include <limits>
#include <thread>
#include <unistd.h>

namespace rocprofsys
{
namespace component
{
namespace
{
using mpip_bundle_t =
    tim::component_tuple<category_region<category::mpi>, comp::comm_data>;

struct comm_rank_data
{
    int       rank = -1;
    int       size = -1;
    uintptr_t comm = mpi_gotcha::null_comm();

    auto updated() const
    {
        return comm != mpi_gotcha::null_comm() && rank >= 0 && size > 0;
    };

    friend bool operator==(const comm_rank_data& _lhs, const comm_rank_data& _rhs)
    {
        auto _lupd = _lhs.updated();
        auto _rupd = _rhs.updated();
        return std::tie(_lupd, _lhs.rank, _lhs.size, _lhs.comm) ==
               std::tie(_rupd, _rhs.rank, _rhs.size, _rhs.comm);
    }

    friend bool operator!=(const comm_rank_data& _lhs, const comm_rank_data& _rhs)
    {
        return !(_lhs == _rhs);
    }

    friend bool operator>(const comm_rank_data& _lhs, const comm_rank_data& _rhs)
    {
        ROCPROFSYS_CI_THROW(!_lhs.updated() && !_rhs.updated(),
                            "Error! comparing rank data that is not updated");

        if(_lhs.updated() && !_rhs.updated()) return true;
        if(!_lhs.updated() && _rhs.updated()) return false;

        if(_lhs.size != _rhs.size) return _lhs.size > _rhs.size;
        if(_lhs.rank != _rhs.rank) return _lhs.rank > _rhs.rank;

        // lesser comm is greater
        return _lhs.comm < _rhs.comm;
    }

    friend bool operator<(const comm_rank_data& _lhs, const comm_rank_data& _rhs)
    {
        return (_lhs != _rhs && !(_lhs > _rhs));
    }
};

uint64_t mpip_index        = std::numeric_limits<uint64_t>::max();
auto     last_comm_record  = comm_rank_data{};
auto     mproc_comm_record = comm_rank_data{};
auto     mpi_comm_records  = std::map<uintptr_t, comm_rank_data>{};

using tim::auto_lock_t;
using tim::type_mutex;

#if defined(TIMEMORY_USE_MPI)
int
rocprofsys_mpi_copy(MPI_Comm, int, void*, void*, void*, int*)
{
    return MPI_SUCCESS;
}

int
rocprofsys_mpi_fini(MPI_Comm, int, void*, void*)
{
    ROCPROFSYS_BASIC_VERBOSE_F(0, "MPI Comm attribute finalize\n");
    auto _blocked = get_sampling_signals();
    if(!_blocked.empty())
        tim::signals::block_signals(_blocked, tim::signals::sigmask_scope::process);
    if(mpip_index != std::numeric_limits<uint64_t>::max())
        comp::deactivate_mpip<mpip_bundle_t, project::rocprofsys>(mpip_index);
    if(is_root_process()) rocprofsys_finalize_hidden();
    return MPI_SUCCESS;
}
#endif

// this ensures rocprofsys_finalize is called before MPI_Finalize
void
rocprofsys_mpi_set_attr()
{
#if defined(TIMEMORY_USE_MPI)
    auto _blocked = get_sampling_signals();
    if(!_blocked.empty())
        tim::signals::block_signals(_blocked, tim::signals::sigmask_scope::process);

    int _comm_key = -1;
    if(PMPI_Comm_create_keyval(&rocprofsys_mpi_copy, &rocprofsys_mpi_fini, &_comm_key,
                               nullptr) == MPI_SUCCESS)
        PMPI_Comm_set_attr(MPI_COMM_SELF, _comm_key, nullptr);

    if(!_blocked.empty())
        tim::signals::unblock_signals(_blocked, tim::signals::sigmask_scope::process);
#endif
    ROCPROFSYS_BASIC_VERBOSE_F(0, "*** Debug:: Configure rocprofsys_mpi_set_attr...\n");
}

using strset_t       = std::set<std::string>;
auto permit_bindings = strset_t{};
auto reject_bindings = strset_t{};
}  // namespace

mpi_gotcha::hash_array_t&
mpi_gotcha::get_hashes()
{
    static auto _v = []() {
        const auto& _data = mpi_gotcha_t::get_gotcha_data();
        auto        _init = hash_array_t{};
        for(size_t i = 0; i < gotcha_capacity; ++i)
        {
            auto&& _id = _data.at(i).tool_id;
            if(!_id.empty())
                _init.at(i) = tim::add_hash_id(_id.c_str());
            else
                ROCPROFSYS_VERBOSE(1, "WARNING!!! mpi_gotcha tool id at index %zu was empty!\n", i);
        }
        return _init;
    }();
    return _v;
}

void
mpi_gotcha::configure()
{
    // don't emit warnings for missing MPI functions unless debug or verbosity >= 3
    if(get_verbose_env() > 0 && !get_debug_env())
    {
        for(size_t i = 0; i < mpi_gotcha_t::capacity(); ++i)
        {
            auto* itr = mpi_gotcha_t::at(i);
            if(itr) itr->verbose = -1;
        }
    }

    mpi_gotcha_t::get_initializer() = []() {
        mpi_gotcha_t::configure(comp::gotcha_config<0, int, int*, char***>{ "MPI_Init" });
        mpi_gotcha_t::configure(comp::gotcha_config<1, int, int*, char***, int, int*>{"MPI_Init_thread" });
        mpi_gotcha_t::configure(comp::gotcha_config<2, int>{ "MPI_Finalize" });
        reject_bindings.emplace("MPI_Init");
        reject_bindings.emplace("MPI_Init_thread");
        reject_bindings.emplace("MPI_Finalize");
#if defined(ROCPROFSYS_USE_MPI_HEADERS) && ROCPROFSYS_USE_MPI_HEADERS > 0
        mpi_gotcha_t::configure(comp::gotcha_config<3, int, comm_t, int*>{ "MPI_Comm_rank" });
        mpi_gotcha_t::configure(comp::gotcha_config<4, int, comm_t, int*>{ "MPI_Comm_size" });
        reject_bindings.emplace("MPI_Comm_rank");
        reject_bindings.emplace("MPI_Comm_size");
#endif
    };

    ROCPROFSYS_BASIC_VERBOSE_F(0, "*** Debug:: Configure MPI wrappers...\n");
    is_disabled();
}

void
mpi_gotcha::shutdown()
{
    ROCPROFSYS_BASIC_VERBOSE_F(0, "*** Debug:: Shutdown MPI wrappers...\n");
    mpi_gotcha_t::disable();
}

mpi_gotcha::mpi_gotcha(const gotcha_data_t& _data)
: m_data{ &_data }
{}

bool
mpi_gotcha::update()
{
    auto_lock_t _lk{ type_mutex<mpi_gotcha>(), std::defer_lock };
    if(!_lk.owns_lock()) _lk.lock();

    comm_rank_data _rank_data = mproc_comm_record;
    for(const auto& itr : mpi_comm_records)
    {
        // skip null comms
        if(itr.first == null_comm()) continue;
        // if currently have null comm, replace
        else if(_rank_data.comm == null_comm())
            _rank_data = itr.second;
        // if
        else if(itr.second > _rank_data)
            _rank_data = itr.second;
    }

    if(_rank_data.updated() && _rank_data != last_comm_record)
    {
        auto _rank = _rank_data.rank;
        auto _size = _rank_data.size;

        tim::mpi::set_rank(_rank);
        tim::mpi::set_size(_size);
        tim::settings::default_process_suffix() = _rank;

        ROCPROFSYS_BASIC_VERBOSE(0, "[pid=%i] MPI rank: %i (%i), MPI size: %i (%i)\n",
                                 process::get_id(), tim::mpi::rank(), _rank,
                                 tim::mpi::size(), _size);
        last_comm_record      = _rank_data;
        config::get_use_pid() = true;
        return true;
    }
    return false;
}

void
mpi_gotcha::disable_comm_intercept()
{
#if defined(ROCPROFSYS_USE_MPI_HEADERS) && ROCPROFSYS_USE_MPI_HEADERS > 0
    mpi_gotcha_t::revert<3>();
    mpi_gotcha_t::revert<4>();
#endif
}

template <typename... Args>
auto
mpi_gotcha::operator()(uintptr_t&&, int (*_callee)(Args...), Args... _args) const
{
    ROCPROFSYS_BASIC_VERBOSE(0, "***Debug:: %s(int*, char***)\n", m_data->tool_id.c_str());

    //rocprofsys_push_trace_hidden(m_data->tool_id.c_str());
#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    tim::mpi::is_initialized_callback() = []() { return true; };
    tim::mpi::is_finalized()            = false;
#endif
    using bundle_t = category_region<category::mpi>;

    if(is_disabled() || m_protect)
    {
        if(!_callee)
        {
            if(m_data)
            {
                ROCPROFSYS_PRINT("Warning! nullptr to %s\n", m_data->tool_id.c_str());
            }
            return MPI_ERR_OTHER;
        }
        return (*_callee)(_args...);
    }

    struct local_dtor
    {
        explicit local_dtor(bool& _v)
        : _protect{ _v }
        {}
        ~local_dtor() { _protect = false; }
        bool& _protect;
    } _dtor{ m_protect = true };

    bundle_t::audit(std::string_view{ m_data->tool_id }, audit::incoming{}, _args...);
    auto _ret = (*_callee)(_args...);
    bundle_t::audit(std::string_view{ m_data->tool_id }, audit::outgoing{}, _ret);

    return _ret;
}

int
mpi_gotcha::operator()(int (*_callee)(int*, char***), int* _argc, char*** _argv) const
{
    ROCPROFSYS_BASIC_VERBOSE(0, "***Debug::%s(int*, char***, int, int*)\n", m_data->tool_id.c_str());

    rocprofsys_push_trace_hidden(m_data->tool_id.c_str());
#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    tim::mpi::is_initialized_callback() = []() { return true; };
    tim::mpi::is_finalized()            = false;
#endif
    if(get_state() != ::rocprofsys::State::Active || m_protect)
        return (*_callee)(_argc, _argv);

    struct local_dtor
    {
        explicit local_dtor(bool& _v) : _protect{ _v } {}
        ~local_dtor() { _protect = false; }
        bool& _protect;
    } _dtor{ m_protect = true };

    //int _retval = (*this)(reinterpret_cast<uintptr_t>(_argc), _callee, _argc, _argv);
    int _retval = (*_callee)(_argc, _argv);

    if(_retval == tim::mpi::success_v && m_data->tool_id.find("MPI_Init") == 0)
    {
        rocprofsys_mpi_set_attr();
        // rocprof-sys will set this environement variable to true in binary rewrite mode
        // when it detects MPI. Hides this env variable from the user to avoid this
        // being activated unwaringly during runtime instrumentation because that
        // will result in double instrumenting the MPI functions (unless the MPI functions
        // were excluded via a regex expression)
        if(get_use_mpip())
        {
            ROCPROFSYS_BASIC_VERBOSE_F(0, "Activating MPI wrappers...\n");

            // use env vars ROCPROFSYS_MPIP_PERMIT_LIST and ROCPROFSYS_MPIP_REJECT_LIST
            // to control the gotcha bindings at runtime
            comp::configure_mpip<mpip_bundle_t, project::rocprofsys>(permit_bindings,
                                                                     reject_bindings);
            mpip_index = comp::activate_mpip<mpip_bundle_t, project::rocprofsys>();
        }

        auto_lock_t _lk{ type_mutex<mpi_gotcha>() };
        if(!mproc_comm_record.updated())
        {
            auto _pid  = getpid();
            auto _ppid = getppid();
            auto _size = mproc::get_concurrent_processes(_ppid).size();
            if(_size > 0)
            {
                mproc_comm_record.comm = _ppid;
                mproc_comm_record.size = m_size = _size;
                auto _rank                      = mproc::get_process_index(_pid, _ppid);
                if(_rank >= 0) mproc_comm_record.rank = m_rank = _rank;
            }
        }
    }
     
    return _retval;
}

int
mpi_gotcha::operator()(int (*_callee)(int*, char***, int, int*), int* _argc, char*** _argv,
                       int _required, int* _provided) const
{
    ROCPROFSYS_BASIC_VERBOSE(0, "****** Debug::%s()\n", m_data->tool_id.c_str());

    auto _blocked = get_sampling_signals();
    if(!_blocked.empty())
        tim::signals::block_signals(_blocked, tim::signals::sigmask_scope::process);

    if(mpip_index != std::numeric_limits<uint64_t>::max())
        comp::deactivate_mpip<mpip_bundle_t, project::rocprofsys>(mpip_index);

#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    tim::mpi::is_initialized_callback() = []() { return false; };
    tim::mpi::is_finalized()            = true;
#else
    if(is_root_process() && rocprofsys::get_state() < rocprofsys::State::Finalized)
        rocprofsys_finalize_hidden();
#endif
    if(get_state() != ::rocprofsys::State::Active || m_protect)
        return (*_callee)(_argc, _argv, _required, _provided);
    // return (*this)(reinterpret_cast<uintptr_t>(_argc), _callee, _argc, _argv, _required, _provided);
    int _retval = (*_callee)(_argc, _argv, _required, _provided);
}

//mpi_gotcha::operator()(int (*_callee)(void)) const
int mpi_gotcha::operator()(int (*_callee)(comm_t, int*), comm_t _comm, int* _val) const
{
    ROCPROFSYS_BASIC_VERBOSE(0, "***Debug::%s()\n", m_data->tool_id.c_str());

    //rocprofsys_push_trace_hidden(_data.tool_id.c_str());
    if(m_data->tool_id == "MPI_Comm_rank")
    {
        m_comm_val = (uintptr_t) _comm;  // NOLINT
        m_rank_ptr = _val;
    }
    else if(m_data->tool_id == "MPI_Comm_size")
    {
        m_comm_val = (uintptr_t) _comm;  // NOLINT
        m_size_ptr = _val;
    }
    else
    {
        ROCPROFSYS_BASIC_PRINT_F("%s(<comm>, %p) :: unexpected function wrapper\n",
                                 m_data->tool_id.c_str(), static_cast<void*>(_val));
    }
    
    if(get_state() != ::rocprofsys::State::Active || m_protect) return (*_callee)(_comm, _val);
    int _retval = (*_callee)(_comm, _val);
    // return (*this)(reinterpret_cast<uintptr_t>(_comm), _callee, _comm, _val);
    if(_retval == tim::mpi::success_v && m_data->tool_id.find("MPI_Comm_") == 0)
    {
        auto_lock_t _lk{ type_mutex<mpi_gotcha>() };
        if(m_comm_val != null_comm())
        {
            auto& _comm_entry = mpi_comm_records[m_comm_val];
            _comm_entry.comm  = m_comm_val;

            auto _get_rank = [&]() {
                return (m_rank_ptr) ? std::max<int>(*m_rank_ptr, m_rank) : m_rank;
            };

            auto _get_size = [&]() {
                return (m_size_ptr) ? std::max<int>(*m_size_ptr, m_size)
                                    : std::max<int>(m_size, _get_rank() + 1);
            };

            if(m_data->tool_id == "MPI_Comm_rank" || m_data->tool_id == "MPI_Comm_size")
            {
                _comm_entry.rank = m_rank = std::max<int>(_comm_entry.rank, _get_rank());
                _comm_entry.size = m_size = std::max<int>(_comm_entry.size, _get_size());
            }
            else
            {
                ROCPROFSYS_BASIC_VERBOSE(
                    0, "%s() returned %i :: unexpected function wrapper\n",
                    m_data->tool_id.c_str(), (int) _retval);
            }

            if(_comm_entry.updated())
            {
                static thread_local int _num_updates = 0;
                static int              _disable_after =
                    tim::get_env<int>("ROCPROFSYS_MPI_MAX_COMM_UPDATES", 4);
                if(_num_updates++ < _disable_after) update();
            }
        }
    }
}

/*
int mpi_gotcha::operator()(int (*_callee)(void)) const
{
    // ROCPROFSYS_BASIC_DEBUG_F("%s() returned %i\n", _data.tool_id.c_str(), (int) _retval);

    if(!settings::use_output_suffix()) settings::use_output_suffix() = true;

    if(_retval == tim::mpi::success_v && m_data->tool_id.find("MPI_Init") == 0)
    {
        rocprofsys_mpi_set_attr();
        // rocprof-sys will set this environement variable to true in binary rewrite mode
        // when it detects MPI. Hides this env variable from the user to avoid this
        // being activated unwaringly during runtime instrumentation because that
        // will result in double instrumenting the MPI functions (unless the MPI functions
        // were excluded via a regex expression)
        if(get_use_mpip())
        {
            ROCPROFSYS_BASIC_VERBOSE_F(2, "Activating MPI wrappers...\n");

            // use env vars ROCPROFSYS_MPIP_PERMIT_LIST and ROCPROFSYS_MPIP_REJECT_LIST
            // to control the gotcha bindings at runtime
            comp::configure_mpip<mpip_bundle_t, project::rocprofsys>(permit_bindings,
                                                                     reject_bindings);
            mpip_index = comp::activate_mpip<mpip_bundle_t, project::rocprofsys>();
        }

        auto_lock_t _lk{ type_mutex<mpi_gotcha>() };
        if(!mproc_comm_record.updated())
        {
            auto _pid  = getpid();
            auto _ppid = getppid();
            auto _size = mproc::get_concurrent_processes(_ppid).size();
            if(_size > 0)
            {
                mproc_comm_record.comm = _ppid;
                mproc_comm_record.size = m_size = _size;
                auto _rank                      = mproc::get_process_index(_pid, _ppid);
                if(_rank >= 0) mproc_comm_record.rank = m_rank = _rank;
            }
        }
    }
    else if(_retval == tim::mpi::success_v && m_data->tool_id.find("MPI_Comm_") == 0)
    {
        auto_lock_t _lk{ type_mutex<mpi_gotcha>() };
        if(m_comm_val != null_comm())
        {
            auto& _comm_entry = mpi_comm_records[m_comm_val];
            _comm_entry.comm  = m_comm_val;

            auto _get_rank = [&]() {
                return (m_rank_ptr) ? std::max<int>(*m_rank_ptr, m_rank) : m_rank;
            };

            auto _get_size = [&]() {
                return (m_size_ptr) ? std::max<int>(*m_size_ptr, m_size)
                                    : std::max<int>(m_size, _get_rank() + 1);
            };

            if(m_data->tool_id == "MPI_Comm_rank" || m_data->tool_id == "MPI_Comm_size")
            {
                _comm_entry.rank = m_rank = std::max<int>(_comm_entry.rank, _get_rank());
                _comm_entry.size = m_size = std::max<int>(_comm_entry.size, _get_size());
            }
            else
            {
                ROCPROFSYS_BASIC_VERBOSE(
                    0, "%s() returned %i :: unexpected function wrapper\n",
                    m_data->tool_id.c_str(), (int) _retval);
            }

            if(_comm_entry.updated())
            {
                static thread_local int _num_updates = 0;
                static int              _disable_after =
                    tim::get_env<int>("ROCPROFSYS_MPI_MAX_COMM_UPDATES", 4);
                if(_num_updates++ < _disable_after) update();
            }
        }
    }
    rocprofsys_pop_trace_hidden(m_data.tool_id.c_str());
    if(get_state() != ::rocprofsys::State::Active || m_protect)
        return (*_callee)(_comm, _val);
    return (*this)(reinterpret_cast<uintptr_t>(_comm), _callee, _comm, _val);
}
*/
bool
mpi_gotcha::is_disabled()
{
    bool disabled = (get_state() != ::rocprofsys::State::Active);
    ROCPROFSYS_BASIC_VERBOSE_F(0, "*** Debug:: MPI wrappers disabled... %s\n",  disabled ? "true" : "false");
    return get_state() != ::rocprofsys::State::Active;
}
}  // namespace component
}  // namespace rocprofsys

TIMEMORY_INITIALIZE_STORAGE(rocprofsys::component::mpi_gotcha)
