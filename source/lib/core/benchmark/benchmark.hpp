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

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "core/benchmark/category.hpp"
#include "core/debug.hpp"

namespace rocprofsys
{
namespace benchmark
{
namespace
{
template <bool enabled, typename category_enum, category_enum... enabled_categories>
struct benchmark_impl
{
    template <category_enum... categories>
    struct scope
    {
        /**
 * @brief Deleted copy constructor to make the scope non-copyable.
 *
 * The scope owns a scoped benchmark interval whose end is invoked in the
 * destructor; copying would lead to double-end semantics or missed measurements,
 * so copying is explicitly disabled.
 */
scope(const scope&)            = delete;
        /**
 * @brief Deleted copy assignment operator to prevent copying of `scope` instances.
 *
 * The `scope` type is non-copyable (move-only); disabling copy-assignment avoids
 * duplicating a scope's lifetime semantics and ensures a single RAII owner.
 */
scope& operator=(const scope&) = delete;
        /**
 * @brief Destroys the scoped benchmark object.
 *
 * The destructor stops the benchmark scope by invoking end for the scope's
 * compile-time categories (releasing the timing region recorded by the
 * corresponding start). This ensures any started timers for the scope's
 * categories are closed when the object goes out of scope.
 */
~scope()                       = default;

    protected:
        /**
 * @brief Default constructs a scoped benchmark object.
 *
 * The constructor begins timing for the scope's compile-time categories; the
 * corresponding timing is ended when the scoped object is destroyed.
 */
scope()                   = default;
        /**
 * @brief Move-constructs a scope.
 *
 * Transfers ownership of the scoped benchmark to the target object.
 * The moved-from object is left in a valid but unspecified state.
 */
scope(scope&&)            = default;
        /**
 * @brief Move-assigns a scope from another instance.
 *
 * Defaulted move assignment operator that transfers ownership/state from the
 * source scope into this one. The source is left in a valid but unspecified state.
 */
scope& operator=(scope&&) = default;
    };

    template <category_enum... categories>
    /**
     * @brief No-op placeholder for benchmarking start when benchmarks are disabled.
     *
     * This function intentionally does nothing. It exists so call sites can invoke
     * a uniform `start()` API regardless of whether benchmarking is enabled at
     * compile time. Has no side effects.
     */
    static void start()
    {}

    template <category_enum... categories>
    /**
     * @brief No-op end point for a benchmark category.
     *
     * When benchmarking is compiled out or disabled, this function provides a
     * compatible API but performs no actions.
     */
    static void end()
    {}

    template <category_enum... categories>
    /**
     * @brief Create an RAII benchmarking scope for the specified categories.
     *
     * Returns a scope instance that begins timing for the given compile-time
     * categories on construction and ends timing on destruction. Use the returned
     * object to measure a lexical scope; the `[[nodiscard]]` attribute indicates
     * the value must be used (otherwise the scope would be destroyed immediately).
     *
     * @return scope<categories...> RAII scope that records start/end for the categories.
     */
    [[nodiscard]] static scope<categories...> scoped_trace()
    {
        return scope<categories...>{};
    }

    /**
 * @brief No-op placeholder that would initialize enabled benchmark categories from an environment variable.
 *
 * When benchmarking is compiled out/disabled, this implementation does nothing.
 *
 * @param envVar Optional name of the environment variable to read (if non-null). When benchmarking is enabled,
 *               the implementation interprets this string as the environment variable that lists comma-separated
 *               categories to enable. Here it is accepted but ignored.
 */
static void init_from_env(const char* = nullptr) {}
    /**
 * @brief Print collected benchmark results for enabled categories.
 *
 * When benchmarking is enabled, prints a sorted, human-readable table of
 * per-category statistics (calls, total time, average, min, max). Only
 * categories with at least one recorded call are shown. Results are derived
 * from the accumulated per-thread timing data and are formatted for console
 * output.
 *
 * This function is safe to call from user code after benchmarks have run;
 * it acquires internal synchronization to read the collected results.
 */
static void show_results() {}
};

using tid_t = __pid_t;
struct indexed_category
{
    size_t category;
    tid_t  thread_id;

    friend /**
     * @brief Compare two indexed_category values for equality.
     *
     * Returns true if both the category and the associated thread id are equal.
     *
     * @param lhs Left-hand indexed_category to compare.
     * @param rhs Right-hand indexed_category to compare.
     * @return true if both category and thread_id match; otherwise false.
     */
    bool operator==(const indexed_category& lhs, const indexed_category& rhs)
    {
        return lhs.category == rhs.category && lhs.thread_id == rhs.thread_id;
    }
};

struct indexed_category_hash
{
    /**
     * @brief Computes a hash for an indexed_category (category + thread id).
     *
     * Combines the hash of the category index and the thread id using XOR and a left shift
     * to produce a single size_t hash value suitable for use in unordered containers.
     *
     * @param p The indexed_category to hash.
     * @return std::size_t Combined hash of p.category and p.thread_id.
     */
    size_t operator()(const indexed_category& p) const noexcept
    {
        std::size_t hash1 = std::hash<size_t>{}(p.category);
        std::size_t hash2 = std::hash<size_t>{}(p.thread_id);
        return hash1 ^ (hash2 << 1);
    }
};

template <typename category_enum, category_enum... enabled_categories>
struct benchmark_impl<true, category_enum, enabled_categories...>
{
    static_assert(std::is_enum_v<category_enum>, "category_enum must be an enum");

public:
    using clock                             = std::chrono::high_resolution_clock;
    using time_point                        = clock::time_point;
    static constexpr size_t _max_categories = static_cast<size_t>(category_enum::count);

    template <category_enum... categories>
    struct scope
    {
        friend benchmark_impl;

    public:
        /**
 * @brief Deleted copy constructor to make the scope non-copyable.
 *
 * The scope owns a scoped benchmark interval whose end is invoked in the
 * destructor; copying would lead to double-end semantics or missed measurements,
 * so copying is explicitly disabled.
 */
scope(const scope&)            = delete;
        /**
 * @brief Deleted copy assignment operator to prevent copying of `scope` instances.
 *
 * The `scope` type is non-copyable (move-only); disabling copy-assignment avoids
 * duplicating a scope's lifetime semantics and ensures a single RAII owner.
 */
scope& operator=(const scope&) = delete;
        /**
 * @brief RAII destructor that ends timing for the scope's categories.
 *
 * Calls end<categories...>() so the benchmark records end timestamps and updates
 * aggregated results for each category when the scope object is destroyed.
 */
~scope() { end<categories...>(); }

    protected:
        /**
 * @brief Creates an RAII scope that starts timing for the specified categories.
 *
 * Records start timestamps for each category in the template parameter pack.
 * The scope's destructor invokes end<categories...>() to record elapsed
 * durations, so this constructor should be paired with the destructor to
 * produce a complete measurement.
 */
scope() { start<categories...>(); }

        /**
 * @brief Move-constructs a scope.
 *
 * Transfers ownership of the scoped benchmark to the target object.
 * The moved-from object is left in a valid but unspecified state.
 */
scope(scope&&)            = default;
        /**
 * @brief Move-assigns a scope from another instance.
 *
 * Defaulted move assignment operator that transfers ownership/state from the
 * source scope into this one. The source is left in a valid but unspecified state.
 */
scope& operator=(scope&&) = default;
    };

    template <category_enum... categories>
    /**
     * @brief Record the start timestamp for each specified benchmark category for the current thread.
     *
     * For every category in the template parameter pack, if that category is enabled in m_enabled,
     * captures the current high-resolution time and stores it in m_started keyed by (category index, thread id).
     * Access to shared state is synchronized via m_mutex. If an entry already exists for the same
     * (category, thread) it is overwritten.
     */
    static void start()
    {
        static const thread_local auto _thread_id = gettid();
        const auto                     now        = clock::now();
        std::lock_guard                lock(m_mutex);
        (..., (is_category_defined<categories>([&] {
             if(m_enabled.test(to_index(categories)))
                 m_started[{ to_index(categories), _thread_id }] = now;
         })));
    }

    template <category_enum... categories>
    /**
     * @brief Record end timestamps for the given benchmark categories and update their accumulated results.
     *
     * For each compile-time category in the template parameter pack, if that category is currently enabled
     * this function looks up the per-thread start time, computes the elapsed microseconds since start,
     * and updates the aggregated result (count, total, min, max). If no matching start time is found for
     * a category/thread pair, a warning is emitted and that category is skipped.
     *
     * Side effects:
     * - Reads a thread-local thread id.
     * - Locks internal mutex to access and modify shared start-time and result storage.
     * - May modify per-category aggregated statistics.
     *
     * Behavior notes:
     * - Categories that are not enabled are ignored.
     * - This function is intended to be called as the counterpart to the corresponding `start<...>()`
     *   call (or implicitly via `scope<...>` destruction).
     */
    static void end()
    {
        static const thread_local auto _thread_id = getpid();
        const auto                     _end_time  = clock::now();
        std::lock_guard                lock(m_mutex);
        (..., (is_category_defined<categories>([&] {
             if(m_enabled.test(to_index(categories)))
                 end_category(_end_time, categories, _thread_id);
         })));
    }

    template <category_enum... categories>
    /**
     * @brief Create an RAII benchmarking scope for the specified categories.
     *
     * Returns a scope instance that begins timing for the given compile-time
     * categories on construction and ends timing on destruction. Use the returned
     * object to measure a lexical scope; the `[[nodiscard]]` attribute indicates
     * the value must be used (otherwise the scope would be destroyed immediately).
     *
     * @return scope<categories...> RAII scope that records start/end for the categories.
     */
    [[nodiscard]] static scope<categories...> scoped_trace()
    {
        return scope<categories...>{};
    }

    /**
     * @brief Initialize enabled benchmark categories from an environment variable.
     *
     * Reads a comma-separated list of category names from the given environment variable
     * (default: "ROCPROFSYS_BENCHMARK_CATEGORIES"), trims whitespace around each token,
     * and enables any matching compiled categories by setting their bits in the internal
     * enabled bitset.
     *
     * This function acquires the internal mutex for the duration of the operation and
     * updates the static m_enabled bitset as a side effect. If the environment variable
     * is absent or empty, a warning is emitted and no categories are enabled.
     *
     * @param envVar Name of the environment variable to read (defaults to
     *               "ROCPROFSYS_BENCHMARK_CATEGORIES").
     */
    static void init_from_env(const char* envVar = "ROCPROFSYS_BENCHMARK_CATEGORIES")
    {
        std::lock_guard lock(m_mutex);
        const auto*     env = std::getenv(envVar);
        if(env == nullptr || std::string(env).empty())
        {
            ROCPROFSYS_WARNING(1, "No BENCHMARK categories specified in environment "
                                  "variable ROCPROFSYS_BENCHMARK_CATEGORIES.\n");
            return;
        }
        std::string        _str(env);
        std::istringstream ss(_str);
        std::string        token;

        while(std::getline(ss, token, ','))
        {
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            for(category_enum cat : compiledCategories)
            {
                if(to_string(cat) == token)
                {
                    m_enabled.set(to_index(cat));
                }
            }
        }
    }

    /**
     * @brief Print collected benchmark results to stdout in a human-readable table.
     *
     * Collects recorded per-category results that have at least one measurement, sorts
     * them by total accumulated time (descending), and writes a formatted table to
     * standard output. Total time is shown in milliseconds, average/min/max are in
     * microseconds. The function takes the internal mutex while reading and formatting
     * data, so it is safe to call concurrently with other benchmarking operations.
     *
     * The output includes colored separators and the following columns:
     * Category | Calls | Total(ms) | Avg(us) | Min(us) | Max(us)
     *
     * This function does not return a value and does not throw exceptions.
     */
    static void show_results()
    {
        std::lock_guard                                    lock(m_mutex);
        std::vector<std::pair<category_enum, result_data>> sorted;

        for(category_enum cat : compiledCategories)
        {
            const auto& data = m_results[to_index(cat)];
            if(data.count > 0)
            {
                sorted.emplace_back(cat, data);
            }
        }

        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.second.total_time > b.second.total_time;
        });

        constexpr uint32_t _category = 30;
        constexpr uint32_t _calls    = 8;
        constexpr uint32_t _total    = 12;
        constexpr uint32_t _avg      = 10;
        constexpr uint32_t _min      = 10;
        constexpr uint32_t _max      = 10;

        std::cout << "\033[32m"
                  << std::string(_category + _calls + _total + _avg + _min + _max, '=')
                  << "\n";
        std::cout << "Benchmark Results (Sorted by Total Time):\n";
        std::cout << std::string(_category + _calls + _total + _avg + _min + _max, '-')
                  << "\n";
        std::cout << std::left << std::setw(_category) << "Category" << std::right
                  << std::setw(_calls) << "Calls" << std::setw(_total) << "Total(ms)"
                  << std::setw(_avg) << "Avg(us)" << std::setw(_min) << "Min(us)"
                  << std::setw(_max) << "Max(us)" << "\n";

        std::cout << std::string(_category + _calls + _total + _avg + _min + _max, '-')
                  << "\n";

        for(const auto& [cat, data] : sorted)
        {
            double totalMs = static_cast<double>(data.total_time) / 1000.0;
            double avgUs   = static_cast<double>(data.total_time) / data.count;

            std::cout << std::left << std::setw(_category) << to_string(cat) << std::right
                      << std::setw(_calls) << data.count << std::setw(_total)
                      << std::fixed << std::setprecision(3) << totalMs << std::setw(_avg)
                      << std::fixed << std::setprecision(1) << avgUs << std::setw(_min)
                      << data.min_time << std::setw(_max) << data.max_time << "\n";
        }

        std::cout << std::string(_category + _calls + _total + _avg + _min + _max, '=')
                  << "\033[0m" << "\n\n";
    }

private:
    struct result_data
    {
        uint64_t total_time = 0;
        size_t   count      = 0;
        uint64_t min_time   = std::numeric_limits<uint64_t>::max();
        uint64_t max_time   = std::numeric_limits<uint64_t>::min();

        /**
         * @brief Incorporates a measured duration into the aggregate statistics.
         *
         * Updates the accumulated total_time and call count, and adjusts min_time and
         * max_time to include the provided measurement.
         *
         * @param duration Measured duration in microseconds to add to the statistics.
         */
        void update(uint64_t duration)
        {
            total_time += duration;
            count += 1;
            if(duration < min_time) min_time = duration;
            if(duration > max_time) max_time = duration;
        }
    };

    /**
     * @brief Convert a category enum value to its zero-based index.
     *
     * Converts the provided category enum to a size_t index via a constexpr cast.
     * The enum's enumerators are expected to correspond to contiguous integer
     * indices suitable for indexing arrays of per-category data.
     *
     * @param cat Category enum value to convert.
     * @return size_t Zero-based index corresponding to `cat`.
     */
    static constexpr size_t to_index(category_enum cat)
    {
        return static_cast<size_t>(cat);
    }

    /**
     * @brief Finalize timing for a single category on a thread and record the elapsed time.
     *
     * Looks up the matching start time for (category, thread_id), computes the elapsed
     * duration in microseconds from that start to end_time, updates the aggregated
     * results for the category, and removes the start entry.
     *
     * If no matching start time is found, logs a warning and does nothing.
     *
     * @param end_time Clock time point representing the end timestamp.
     * @param cat Category whose timing is being ended.
     * @param thread_id Thread identifier used to locate the corresponding start time.
     */
    static void end_category(const time_point& end_time, category_enum cat,
                             const tid_t thread_id)
    {
        const size_t _idx = to_index(cat);
        auto         _it  = m_started.find({ _idx, thread_id });
        if(_it == m_started.end())
        {
            ROCPROFSYS_WARNING(1, "Benchmark error: missing start time for category!\n");
            return;
        }

        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - _it->second)
                .count();
        m_started.erase(_it);
        m_results[_idx].update(duration);
    }

    template <category_enum Cat, typename Func>
    /**
     * @brief Invoke a callback only when a compile-time category is enabled.
     *
     * Conditionally calls the provided callable `f` if and only if the compile-time
     * category `Cat` is present in the list of enabled_categories for this
     * specialization. The check is performed with `if constexpr`, so when the
     * category is not enabled `f` is not instantiated or evaluated.
     *
     * @tparam Cat The category to test (compile-time enum value).
     * @tparam Func Callable type; must be invocable with no arguments when `Cat` is enabled.
     * @param f Callable that will be invoked only when `Cat` is among enabled_categories.
     */
    static constexpr void is_category_defined(Func&& f)
    {
        if constexpr(((Cat == enabled_categories) || ...))
        {
            f();
        }
    }

    static constexpr std::array<category_enum, sizeof...(enabled_categories)>
        compiledCategories = { enabled_categories... };

    static inline std::unordered_map<indexed_category, time_point, indexed_category_hash>
                                                           m_started;
    static inline std::array<result_data, _max_categories> m_results{};
    static inline std::bitset<_max_categories>             m_enabled;
    static inline std::mutex                               m_mutex;
};

#ifdef ROCPROFSYS_ENABLE_BENCHMARK
using _benchmark_impl = benchmark::benchmark_impl<
    static_cast<bool>(ROCPROFSYS_ENABLE_BENCHMARK), benchmark::category,
    benchmark::category::kernel_dispatch, benchmark::category::memory_copy,
    benchmark::category::memory_allocate, benchmark::category::db_entry_kernel_dispatch,
    benchmark::category::db_entry_memory_copy,
    benchmark::category::db_entry_memory_allocate,
    benchmark::category::perfetto_kernel_dispatch,
    benchmark::category::sdk_tool_buffered_tracing>;
#else
using _benchmark_impl = benchmark::benchmark_impl<false, benchmark::category>;
#endif
}  // namespace

template <category... categories>
/**
 * @brief Start timing for the specified benchmark categories.
 *
 * Delegates to the internal benchmark implementation to record start timestamps
 * for the categories in the surrounding template parameter pack. When the
 * benchmarking facility is disabled this function is a no-op.
 */
void
start()
{
    _benchmark_impl::template start<categories...>();
}

template <category... categories>
/**
 * @brief Stop timing for the compile-time category pack.
 *
 * Ends the benchmark measurement for each category in the template parameter pack and records the elapsed
 * duration (per-thread) into the accumulated results. This is a thin wrapper that forwards to the
 * underlying benchmark implementation.
 *
 * If a matching start was not recorded for a category/thread, a warning will be issued and that category
 * is not updated.
 */
void
end()
{
    _benchmark_impl::template end<categories...>();
}

template <category... categories>
/**
 * @brief Create an RAII benchmark scope for the specified categories.
 *
 * Returns an object that records start timestamps for the given benchmark
 * categories on construction and records corresponding end timestamps when
 * the object is destroyed. The return value is marked [[nodiscard]] — keep
 * the scope object alive for the duration you want measured.
 *
 * @return scope object that pairs start/end for the specified categories.
 */
[[nodiscard]] auto
scoped_trace()
{
    return _benchmark_impl::template scoped_trace<categories...>();
}

/**
 * @brief Initialize enabled benchmark categories from an environment variable.
 *
 * Reads a comma-separated list of category names from the environment variable named by `envVar`
 * (defaults to "BENCHMARK_CATEGORIES") and enables matching benchmark categories for result
 * collection. Matching is performed against the compiled category names; unknown or empty values
 * are ignored and will produce a runtime warning.
 *
 * @param envVar Name of the environment variable to read (null-terminated C string).
 */
inline void
init_from_env(const char* envVar = "BENCHMARK_CATEGORIES")
{
    _benchmark_impl::init_from_env(envVar);
}

/**
 * @brief Print collected benchmark results.
 *
 * Prints a formatted table of benchmark results for enabled categories (calls, total
 * time, average, min, max). Results are produced from previously paired start/end
 * calls or scoped traces; categories with zero calls are omitted. The output is
 * generated on the process' standard output.
 */
inline void
show_results()
{
    _benchmark_impl::show_results();
}

}  // namespace benchmark
}  // namespace rocprofsys
