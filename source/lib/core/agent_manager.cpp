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

#include "agent_manager.hpp"
#include "debug.hpp"
#include <algorithm>
#include <iterator>

namespace rocprofsys
{

/**
 * @brief Return the global singleton instance of agent_manager.
 *
 * The instance is lazily initialized on first call and is safe for concurrent
 * use under the C++11 (and later) static initialization guarantees.
 *
 * @return Reference to the single agent_manager instance.
 */
agent_manager&
agent_manager::get_instance()
{
    static agent_manager instance;
    return instance;
}

/**
 * @brief Insert an agent into the manager and assign it a device ID.
 *
 * Assigns a new sequential device_id to the provided agent based on its type
 * (GPU agents use the internal GPU counter, CPU agents use the CPU counter),
 * increments the corresponding counter, and stores an internal shared copy of
 * the agent in the manager's collection.
 *
 * @param _agent Agent to insert. The agent's `device_id` is updated in-place
 *               to the assigned value; a separate shared copy of the agent is
 *               stored internally.
 */
void
agent_manager::insert_agent(agent& _agent)
{
    ROCPROFSYS_VERBOSE(
        3, "Inserting agent with device handle: %lu, and agent id: %ld, device type: %s",
        _agent.device_id,
        (_agent.type == agent_type::GPU ? _gpu_agents_cnt : _cpu_agents_cnt),
        (_agent.type == agent_type::GPU ? "GPU" : "CPU"));

    _agent.device_id =
        (_agent.type == agent_type::GPU ? _gpu_agents_cnt++ : _cpu_agents_cnt++);
    _agents.emplace_back(std::make_shared<agent>(_agent));
}

/**
 * @brief Retrieve a const reference to the agent with the given per-type device id.
 *
 * Searches the manager's stored agents for an entry whose type matches `type`
 * and whose device_id equals `device_id`. Returns a const reference to the found
 * agent.
 *
 * @param device_id The per-type device identifier previously assigned to the agent.
 * @param type The agent_type to match (e.g., GPU or CPU).
 * @return const agent& Reference to the matching agent.
 *
 * @throws std::out_of_range If no agent with the specified type and device_id exists.
 */
const agent&
agent_manager::get_agent_by_id(size_t device_id, agent_type type) const
{
    ROCPROFSYS_VERBOSE(3, "Getting agent for device id: %ld, type %s\n", device_id,
                       (type == agent_type::GPU) ? "GPU" : "CPU");
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->type == type && agent_ptr->device_id == device_id;
        });
    if(_agent == _agents.end())
    {
        std::ostringstream oss;
        oss << "Agent not found for device id: " << device_id
            << ", type: " << (type == agent_type::GPU ? "GPU" : "CPU");
        throw std::out_of_range(oss.str());
    }
    return **_agent;
}

/**
 * @brief Retrieve a const reference to an agent by its device handle and type.
 *
 * Searches the manager's agent collection for an agent whose `type` equals `type`
 * and whose `id` equals `device_handle`. If found, returns a const reference
 * to that agent.
 *
 * @param device_handle The agent's device handle (stored in agent::id) to match.
 * @param type The agent_type the agent must have (e.g., GPU or CPU).
 * @return const agent& Reference to the matching agent.
 * @throws std::out_of_range If no agent with the given handle and type exists.
 */
const agent&
agent_manager::get_agent_by_handle(uint64_t device_handle, agent_type type) const
{
    ROCPROFSYS_VERBOSE(3, "Getting agent for device handle: %ld, type %s\n",
                       device_handle, (type == agent_type::GPU ? "GPU" : "CPU"));
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->type == type && agent_ptr->id == device_handle;
        });
    if(_agent == _agents.end())
    {
        std::ostringstream oss;
        oss << "Agent not found for device handle: " << device_handle
            << ", type: " << (type == agent_type::GPU ? "GPU" : "CPU");
        throw std::out_of_range(oss.str());
    }
    return **_agent;
}

/**
 * @brief Retrieve an agent by its device handle (agent::id).
 *
 * Searches stored agents for one whose id equals the provided device_handle and returns a const reference to it.
 *
 * @param device_handle Device handle corresponding to agent::id.
 * @return const agent& Reference to the matching agent.
 * @throws std::out_of_range If no agent with the given handle exists.
 */
const agent&
agent_manager::get_agent_by_handle(size_t device_handle) const
{
    ROCPROFSYS_VERBOSE(3, "Getting agent for device handle: %ld\n", device_handle);
    auto _agent =
        std::find_if(_agents.begin(), _agents.end(), [&](const auto& agent_ptr) {
            return agent_ptr->id == device_handle;
        });
    if(_agent == _agents.end())
    {
        std::ostringstream oss;
        oss << "Agent not found for device handle: " << device_handle;
        throw std::out_of_range(oss.str());
    }
    return **_agent;
}

/**
 * @brief Return all managed agents matching the specified device type.
 *
 * Returns a vector of shared_ptrs to the agents whose agent::type equals the
 * provided type. The returned shared_ptrs share ownership of the same agent
 * objects stored by the manager.
 *
 * @param type The agent_type to filter by (e.g., GPU or CPU).
 * @return std::vector<std::shared_ptr<agent>> Vector of shared_ptrs to matching agents;
 *         empty if none match.
 */
std::vector<std::shared_ptr<agent>>
agent_manager::get_agents_by_type(agent_type type) const
{
    ROCPROFSYS_VERBOSE(3, "Getting agent for device type: %s\n",
                       type == agent_type::GPU ? "GPU" : "CPU");

    std::vector<std::shared_ptr<agent>> agents;
    std::copy_if(std::begin(_agents), std::end(_agents), std::back_inserter(agents),
                 [&type](const auto& agent_ptr) { return agent_ptr->type == type; });
    return agents;
}

/**
 * @brief Returns the collection of managed agents.
 *
 * The caller receives a copy of the internal vector of shared pointers to agents.
 * The shared_ptrs preserve shared ownership of the agent objects; modifying
 * the returned vector does not affect the manager's container, but mutating the
 * agent objects through the shared_ptrs will affect the shared instances.
 *
 * @return std::vector<std::shared_ptr<agent>> Copy of the manager's agent list.
 */
std::vector<std::shared_ptr<agent>>
agent_manager::get_agents() const
{
    return _agents;
}

/**
 * @brief Returns the number of registered GPU agents.
 *
 * @return size_t Count of GPU agents tracked by the manager.
 */
size_t
agent_manager::get_gpu_agents_count() const
{
    return _gpu_agents_cnt;
}

/**
 * @brief Returns the number of CPU agents tracked by the manager.
 *
 * @return size_t Count of CPU agents (_cpu_agents_cnt).
 */
size_t
agent_manager::get_cpu_agents_count() const
{
    return _cpu_agents_cnt;
}

}  // namespace rocprofsys
