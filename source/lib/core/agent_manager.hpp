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
#include <cstddef>
#include <memory>
#include <vector>

#include "agent.hpp"

namespace rocprofsys
{

struct agent_manager
{
    static agent_manager& get_instance();

    /**
 * @brief Deleted copy constructor to make agent_manager non-copyable.
 *
 * Ensures there can be no copied instances of agent_manager; access the single instance via get_instance().
 */
agent_manager(const agent_manager&)            = delete;
    /**
 * @brief Deleted copy assignment operator to prevent copying of the singleton-like manager.
 *
 * The agent_manager is non-copyable; assignments are explicitly disabled to preserve
 * unique ownership and internal state consistency.
 */
agent_manager& operator=(const agent_manager&) = delete;
    /**
 * @brief Deleted move constructor to prohibit moving the singleton-like manager.
 *
 * The agent_manager is non-movable to ensure a single, stable instance returned by
 * get_instance(). Attempting to move an agent_manager is ill-formed.
 */
agent_manager(agent_manager&&)                 = delete;
    /**
 * @brief Deleted move-assignment operator.
 *
 * Prevents move-assignment of agent_manager to enforce a single, non-movable instance.
 */
agent_manager& operator=(agent_manager&&)      = delete;
    /**
 * @brief Default destructor.
 *
 * Destroys the agent_manager and releases ownership of all managed agents.
 */
~agent_manager()                               = default;

    void         insert_agent(agent& agent);
    const agent& get_agent_by_id(size_t device_id, agent_type type) const;
    const agent& get_agent_by_handle(size_t device_id, agent_type type) const;
    const agent& get_agent_by_handle(size_t device_handle) const;

    std::vector<std::shared_ptr<agent>> get_agents_by_type(agent_type type) const;

    std::vector<std::shared_ptr<agent>> get_agents() const;

    size_t get_gpu_agents_count() const;
    size_t get_cpu_agents_count() const;

private:
    std::vector<std::shared_ptr<agent>> _agents;
    size_t                              _gpu_agents_cnt{ 0 };
    size_t                              _cpu_agents_cnt{ 0 };
    /**
 * @brief Default constructs an agent_manager.
 *
 * Exists as a private, defaulted constructor to prevent public instantiation;
 * use agent_manager::get_instance() to access the singleton-like manager.
 */
agent_manager() = default;
};

}  // namespace rocprofsys
