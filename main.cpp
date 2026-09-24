#include <iostream>

#include "DataProc.h"
#include "AdaptiveSearch.h"
#include "GreedySearch.h"
#include "GuidedSearch.h"
#include "RandomSearch.h"
#include "TabuSearch.h"
#include "Schedule.h"
#include "Job.h"
#include "Machine.h"
#include "Order.h"
#include "Experiment.h"
#include "CoptInitialSolution.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <random>
#include <string.h>
#include <vector>

using namespace std;

namespace {

constexpr int kMaximumOuterCalls = 1200;
constexpr int kMaximumEliteArchiveSize = 8;

bool GuidedSearchRequested() {
    const char *value = std::getenv("HLS_ENABLE_GUIDED_SEARCH");
    return value == nullptr || std::string(value) != "0";
}

bool GuidedSearchForced() {
    const char *value = std::getenv("HLS_ENABLE_GUIDED_SEARCH");
    return value != nullptr && std::string(value) == "force";
}

int GetMethodIterationBudget(
    const SearchDecision &decision,
    const int configured_iterations,
    const double remaining_fraction
) {
    if (configured_iterations <= 0) {
        return 0;
    }
    if (decision.method == SearchMethod::Greedy) {
        const int normal_budget = std::max(1, std::min(
            configured_iterations,
            std::max(2, (configured_iterations + 4) / 5)
        ));
        const int budget = decision.intensification
            ? std::min(configured_iterations, 3)
            : normal_budget;
        return remaining_fraction <= 0.10 ? std::min(budget, 2) : budget;
    }
    if (decision.method == SearchMethod::Tabu) {
        const int budget = decision.intensification
            ? std::min(configured_iterations, 15)
            : configured_iterations;
        return remaining_fraction <= 0.10 ? std::min(budget, 10) : budget;
    }
    if (decision.method == SearchMethod::Guided) {
        return remaining_fraction <= 0.10
            ? std::min(configured_iterations, 10)
            : configured_iterations;
    }
    return configured_iterations;
}

double GetMethodTimeBudgetSeconds(
    const SearchDecision &decision,
    const double remaining_seconds,
    const double remaining_fraction
) {
    double budget = 0.0;
    if (decision.method == SearchMethod::Greedy) {
        budget = decision.intensification ? 3.0 : 10.0;
    } else if (decision.method == SearchMethod::Tabu) {
        budget = decision.intensification ? 10.0 : 20.0;
    } else if (decision.method == SearchMethod::Guided) {
        budget = 20.0;
    } else {
        budget = 30.0;
    }

    if (remaining_fraction <= 0.10) {
        budget = std::min(
            budget,
            decision.method == SearchMethod::Greedy ? 2.0 : 5.0
        );
    }
    return std::max(0.0, std::min(budget, remaining_seconds));
}

SearchDeadline BuildCallDeadline(
    const SearchDeadline optimization_deadline,
    const double call_budget_seconds
) {
    const SearchDeadline call_deadline = SearchClock::now() +
        std::chrono::duration_cast<SearchClock::duration>(
            std::chrono::duration<double>(call_budget_seconds)
        );
    return std::min(optimization_deadline, call_deadline);
}

Schedule RunSearchMethod(
    const SearchDecision &decision,
    const Schedule &start_schedule,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    const std::vector<std::string> &job_list,
    const int iteration_budget,
    const int population_size,
    const int tabu_list_length,
    const std::vector<Schedule> &elite_archive,
    const SearchDeadline deadline
) {
    switch (decision.method) {
        case SearchMethod::Greedy:
            return GreedySearch(
                start_schedule,
                jobs,
                job_list,
                iteration_budget,
                machines,
                deadline
            );
        case SearchMethod::Genetic:
            return AggressiveSearch(
                start_schedule,
                jobs,
                machines,
                job_list,
                population_size,
                iteration_budget,
                deadline
            );
        case SearchMethod::Tabu:
            return TabuSearch(
                start_schedule,
                jobs,
                job_list,
                tabu_list_length,
                iteration_budget,
                machines,
                deadline
            );
        case SearchMethod::Guided:
            return GuidedInsertionSearch(
                start_schedule,
                elite_archive,
                jobs,
                job_list,
                machines,
                iteration_budget,
                deadline
            );
    }
    return start_schedule;
}

void AddEliteSchedule(
    std::vector<Schedule> &elite_archive,
    const Schedule &schedule
) {
    const std::string signature = ScheduleToString(schedule);
    for (const auto &elite : elite_archive) {
        if (ScheduleToString(elite) == signature) {
            return;
        }
    }
    elite_archive.push_back(schedule);
    if (static_cast<int>(elite_archive.size()) > kMaximumEliteArchiveSize) {
        elite_archive.erase(elite_archive.begin());
    }
}

Schedule BuildMultiStartHeuristicInitialSolution(
    const std::vector<std::string> &job_list,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    const unsigned int random_seed,
    const SearchDeadline deadline
) {
    Schedule best_schedule = GenerateInitialSolution(job_list, jobs, machines);
    if (best_schedule.get_graph().empty()) {
        return best_schedule;
    }

    // Randomized job orders expose different machine bottlenecks while the
    // EET constructor keeps every candidate precedence-feasible.
    const int start_count = std::min(
        32,
        std::max(8, static_cast<int>(jobs.size()) * 2)
    );
    std::mt19937 random_engine(random_seed ^ 0xD1B54A32U);
    std::vector<std::string> shuffled_job_list = job_list;
    for (int start = 1; start < start_count; ++start) {
        if (SearchDeadlineReached(deadline)) {
            break;
        }
        std::shuffle(shuffled_job_list.begin(), shuffled_job_list.end(), random_engine);
        Schedule candidate = GenerateInitialSolution(
            shuffled_job_list,
            jobs,
            machines
        );
        if (!candidate.get_graph().empty() &&
            candidate.get_TotalTime() < best_schedule.get_TotalTime()) {
            best_schedule = std::move(candidate);
        }
    }
    return best_schedule;
}

}  // namespace

int main(int argc, char *argv[]) {
    auto machines = std::vector<Machine>();      // 机器列表
    auto jobs = std::vector<Job>();              // 工件列表
    auto orders = std::vector<Order>();          // 订单列表
    std::string file_path = argc > 1 ? argv[1] : "/data/luopw/production_scheduling/input.txt"; // 运行命令示例：./main input.txt
    // cout << "file_path: " << file_path << endl;
    Init(machines, jobs, orders, file_path);    // 初始化机器、工件、订单
    auto jobList = std::vector<std::string>();   // 用于记录每个工件列表，根据订单生成
    OrderToJobList(orders, jobList);         // 将订单转换为待加工的工件列表
    Schedule input_schedule;                    // 输入的调度方案
    Schedule output_schedule;                   // 输出的调度方案

    // ======parameters========
    // double search_mode_param = 0.5;  // 搜索模式参数0~1，配置选择随机或者贪心
    // double random_search_strategy_param = 0.5;  // 搜索模式参数，配置选择遗传算法或者禁忌搜索
    // int max_iter_count = 30;  // 单轮优化最大迭代次数，表示每次搜索的最大迭代次数
    // int population_size = 20;  // 种群大小，表示遗传算法中种群的大小
    // int tabu_list_length = 10;  // 禁忌表长度，表示禁忌搜索算法中禁忌表的长度
    // int max_repeat_count = 20;  // 最优解最大重复次数，达到最大重复次数则退出搜索

    // double search_mode_param = 0.5;            // 搜索模式参数0~1，配置选择随机或者贪心
    // double random_search_strategy_param = 0.5; // 搜索模式参数，配置选择遗传算法或者禁忌搜索
    // int max_repeat_count = 100;                 // 最优解最大重复次数，达到最大重复次数则退出搜索
    // int max_iter_count = 30;                   // 单轮优化最大迭代次数，表示每次搜索的最大迭代次数
    // int population_size = 20;                  // 种群大小，表示遗传算法中种群的大小
    // int tabu_list_length = 50;                 // 禁忌表长度，表示禁忌搜索算法中禁忌表的长度
    

    double search_mode_param = argc > 2 ? std::stod(argv[2]) : 0.5;            // 搜索模式参数0~1，配置选择随机或者贪心 0.5
    double random_search_strategy_param = argc > 3 ? std::stod(argv[3]) : 0.5; // 搜索模式参数，配置选择遗传算法或者禁忌搜索 0.5
    int max_repeat_count = argc > 4 ? std::stoi(argv[4]) : 100; // 最优解最大重复次数，达到最大重复次数则退出搜索   100
    int max_iter_count = argc > 5 ? std::stoi(argv[5]) : 30;   // 单轮优化最大迭代次数，表示每次搜索的最大迭代次数  30
    int population_size = argc > 6 ? std::stoi(argv[6]) : 20;  // 种群大小，表示遗传算法中种群的大小  20
    int tabu_list_length = argc > 7 ? std::stoi(argv[7]) : 25; // 禁忌表长度，表示禁忌搜索算法中禁忌表的长度 25
    double copt_time_limit_seconds = argc > 8 ? std::stod(argv[8]) : 300.0; // COPT初始解时间限制（秒）
    int total_time_limit_seconds = argc > 9 ? std::stoi(argv[9]) : 3600; // COPT和HLS总优化时间限制（秒）
    unsigned int random_seed = argc > 10
        ? static_cast<unsigned int>(std::stoul(argv[10]))
        : static_cast<unsigned int>(time(nullptr)); // 随机种子
    int operation_count = 0;
    for (const auto &job : jobs) {
        operation_count += job.get_process_count();
    }
    const bool guided_search_enabled = GuidedSearchRequested() &&
        (GuidedSearchForced() || operation_count >= 80);

    if (total_time_limit_seconds <= 0) {
        std::cerr << "total_time_limit_seconds must be positive." << std::endl;
        return 1;
    }

    // ==========================

    cout << "file_path: " << file_path << endl;
    cout << "search_mode_param: " << search_mode_param << endl;
    cout << "random_search_strategy_param: " << random_search_strategy_param << endl;
    cout << "max_repeat_count: " << max_repeat_count << endl;
    cout << "max_iter_count: " << max_iter_count << endl;
    cout << "population_size: " << population_size << endl;
    cout << "tabu_list_length: " << tabu_list_length << endl;
    cout << "copt_time_limit_seconds: " << copt_time_limit_seconds << endl;
    cout << "total_time_limit_seconds: " << total_time_limit_seconds << endl;
    cout << "random_seed: " << random_seed << endl;
    cout << "guided_search_enabled: "
         << (guided_search_enabled ? "true" : "false") << endl;
    cout << "operation_count: " << operation_count << endl;

    // if(argc > 2) {
    //     Schedule schedule0;
    //     if(strcmp(argv[2], "FIFO_EET") == 0) 
    //     {
    //         schedule0 = FIFO_EET(jobs, machines, jobList);
    //     } 
    //     else if(strcmp(argv[2], "FIFO_SPT") == 0) {
    //         schedule0 = FIFO_SPT(jobs, machines, jobList);
    //     }
    //     else if (strcmp(argv[2], "MOPNR_EET") == 0)
    //     {
    //         schedule0 = MOPNR_EET(jobs, machines, jobList);
    //     }
    //     else if (strcmp(argv[2], "MOPNR_SPT") == 0)
    //     {
    //         schedule0 = MOPNR_SPT(jobs, machines, jobList);
    //     }
    //     else if (strcmp(argv[2], "MWKR_EET") == 0)
    //     {
    //         schedule0 = MWKR_EET(jobs, machines, jobList);
    //     }
    //     else if (strcmp(argv[2], "MWKR_SPT") == 0)
    //     {
    //         schedule0 = MWKR_SPT(jobs, machines, jobList);
    //     }
    //     else
    //     {
    //         cout << "Invalid algorithm name. Please use FIFO, MOPNR, MWKR or SPT." << endl;
    //         return 1;
    //     }
    //     // schedule0.to_string();
    //     // schedule0.ToMapCode();
    //     const int ans0 = CalculateTotalTime(schedule0);
    //     cout << "Total time: " << ans0 << endl;

        
    //     // schedule0 = FIFO_EET(jobs, machines, jobList);
    //     // cout << "FIFO_EET Total time: " << CalculateTotalTime(schedule0) << endl;
    //     // schedule0 = FIFO_SPT(jobs, machines, jobList);
    //     // cout << "FIFO_SPT Total time: " << CalculateTotalTime(schedule0) << endl;
    //     // schedule0 = MOPNR_EET(jobs, machines, jobList);
    //     // cout << "MOPNR_EET Total time: " << CalculateTotalTime(schedule0) << endl;
    //     // schedule0 = MOPNR_SPT(jobs, machines, jobList);
    //     // cout << "MOPNR_SPT Total time: " << CalculateTotalTime(schedule0) << endl;
    //     // schedule0 = MWKR_EET(jobs, machines, jobList);
    //     // cout << "MWKR_EET Total time: " << CalculateTotalTime(schedule0) << endl;
    //     // schedule0 = MWKR_SPT(jobs, machines, jobList);
    //     // cout << "MWKR_SPT Total time: " << CalculateTotalTime(schedule0) << endl;
    //     // cout << endl;
    //     return 0;
    // }

    // 总优化计时从 COPT 初始解开始，确保 COPT 时间包含在总时长内。
    const int optimization_start_time = time(nullptr);
    const SearchDeadline optimization_started_at = SearchClock::now();
    const SearchDeadline optimization_deadline = optimization_started_at +
        std::chrono::seconds(total_time_limit_seconds);
    const double copt_time_budget = std::min(
        copt_time_limit_seconds,
        static_cast<double>(total_time_limit_seconds)
    );

    // 使用 COPT 直接从当前 FJSP 实例生成 HLS 初始解。
    Schedule schedule0;
    bool used_copt_initial_solution = true;
    try {
        schedule0 = GenerateCoptInitialSolution(
            jobList,
            jobs,
            machines,
            copt_time_budget
        );
    } catch (const std::exception &exception) {
        used_copt_initial_solution = false;
        std::cerr << "COPT initial solution unavailable: "
                  << exception.what() << std::endl;
        std::cerr << "Falling back to the HLS heuristic initial solution."
                  << std::endl;
        try {
            schedule0 = BuildMultiStartHeuristicInitialSolution(
                jobList,
                jobs,
                machines,
                random_seed,
                optimization_deadline
            );
            if (schedule0.get_graph().empty()) {
                throw std::runtime_error(
                    "HLS heuristic initial solution has an empty graph."
                );
            }
        } catch (const std::exception &fallback_exception) {
            std::cerr << "Failed to generate fallback initial solution: "
                      << fallback_exception.what() << std::endl;
            return 1;
        }
    }

    // schedule0.to_string();
    // schedule0.ToMapCode();
    const int ans0 = CalculateTotalTime(schedule0);
    if (used_copt_initial_solution) {
        cout << "COPT initial makespan: " << ans0 << endl;
    } else {
        cout << "HLS fallback initial makespan: " << ans0 << endl;
    }
    cout << "Total time: " << ans0 << endl;

    input_schedule = schedule0;
    output_schedule = schedule0;

    AdaptiveSearchController search_controller(
        search_mode_param,
        random_search_strategy_param,
        random_seed ^ 0xA511E9B3U,
        guided_search_enabled
    );
    SetRandomSearchSeed(random_seed ^ 0x9E3779B9U);
    SetTabuSearchSeed(random_seed ^ 0x85EBCA6BU);
    SetGuidedSearchSeed(random_seed ^ 0xC2B2AE35U);
    std::vector<Schedule> elite_archive{schedule0};
    int count = 0;
    int stagnation_call_count = 0;
    int no_improvement_round_count = 0;
    std::array<bool, 4> method_failed_since_improvement{};
    method_failed_since_improvement[
        static_cast<int>(SearchMethod::Guided)
    ] = !guided_search_enabled;

    while (count <= kMaximumOuterCalls &&
           no_improvement_round_count <= max_repeat_count &&
           !SearchDeadlineReached(optimization_deadline)) {
        const double remaining_seconds = std::max(
            0.0,
            std::chrono::duration<double>(
                optimization_deadline - SearchClock::now()
            ).count()
        );
        const double remaining_fraction = std::clamp(
            remaining_seconds / total_time_limit_seconds,
            0.0,
            1.0
        );
        const SearchDecision decision = search_controller.Select(
            stagnation_call_count,
            remaining_fraction
        );
        const int iteration_budget = GetMethodIterationBudget(
            decision,
            max_iter_count,
            remaining_fraction
        );
        if (iteration_budget <= 0) {
            break;
        }

        const Schedule *method_start_schedule = &input_schedule;
        if (decision.diversified_start && elite_archive.size() > 1) {
            method_start_schedule = &elite_archive[
                search_controller.SelectArchiveIndex(
                    static_cast<int>(elite_archive.size())
                )
            ];
        }
        const double call_budget_seconds = GetMethodTimeBudgetSeconds(
            decision,
            remaining_seconds,
            remaining_fraction
        );
        const SearchDeadline call_deadline = BuildCallDeadline(
            optimization_deadline,
            call_budget_seconds
        );
        const int global_before_makespan = output_schedule.get_TotalTime();
        const int local_before_makespan = method_start_schedule->get_TotalTime();
        const SearchDeadline call_started_at = SearchClock::now();
        Schedule candidate_schedule = RunSearchMethod(
            decision,
            *method_start_schedule,
            jobs,
            machines,
            jobList,
            iteration_budget,
            population_size,
            tabu_list_length,
            elite_archive,
            call_deadline
        );
        const double elapsed_seconds = std::chrono::duration<double>(
            SearchClock::now() - call_started_at
        ).count();
        const int local_after_makespan = candidate_schedule.get_TotalTime();
        const bool improved = local_after_makespan < global_before_makespan;
        if (improved) {
            output_schedule = std::move(candidate_schedule);
            input_schedule = output_schedule;
            AddEliteSchedule(elite_archive, output_schedule);
            stagnation_call_count = 0;
            no_improvement_round_count = 0;
            method_failed_since_improvement.fill(false);
            method_failed_since_improvement[
                static_cast<int>(SearchMethod::Guided)
            ] = !guided_search_enabled;
            if (!decision.intensification) {
                search_controller.ScheduleIntensification();
            }
        } else {
            if (local_after_makespan == global_before_makespan &&
                ScheduleToString(candidate_schedule) !=
                    ScheduleToString(input_schedule)) {
                input_schedule = candidate_schedule;
                AddEliteSchedule(elite_archive, input_schedule);
            }
            ++stagnation_call_count;
            method_failed_since_improvement[
                static_cast<int>(decision.method)
            ] = true;
            if (std::all_of(
                    method_failed_since_improvement.begin(),
                    method_failed_since_improvement.end(),
                    [](const bool failed) { return failed; }
                )) {
                ++no_improvement_round_count;
                method_failed_since_improvement.fill(false);
                method_failed_since_improvement[
                    static_cast<int>(SearchMethod::Guided)
                ] = !guided_search_enabled;
            }
        }
        search_controller.Record(
            decision.method,
            local_before_makespan,
            local_after_makespan,
            global_before_makespan,
            output_schedule.get_TotalTime(),
            elapsed_seconds
        );

        cout << SearchMethodName(decision.method) << " time: "
             << fixed << setprecision(3) << elapsed_seconds
             << defaultfloat << "s"
             << ", mode: "
             << (decision.intensification ? "intensification" :
                 (decision.diversified_start ? "diversified" : "adaptive"))
             << ", iterations: " << iteration_budget << endl;
        cout << "Total time: " << output_schedule.get_TotalTime() << endl;
        ++count;
    }

    std::vector<SearchMethod> summary_methods{
        SearchMethod::Greedy,
        SearchMethod::Genetic,
        SearchMethod::Tabu
    };
    if (guided_search_enabled) {
        summary_methods.push_back(SearchMethod::Guided);
    }
    for (const SearchMethod method : summary_methods) {
        const SearchMethodStats &stats = search_controller.GetStats(method);
        cout << SearchMethodName(method)
             << " summary: calls=" << stats.calls
             << ", local_improvements=" << stats.improvements
             << ", global_improvements=" << stats.global_improvements
             << ", relative_gain=" << fixed << setprecision(6)
             << stats.total_relative_gain
             << ", time=" << setprecision(3) << stats.total_seconds
             << defaultfloat << "s" << endl;
    }

    // output_schedule.to_string();
    // output_schedule.ToMapCode();
    const int ans1 = CalculateTotalTime(output_schedule);
    // cout << "Total time: " << ans1 << endl;
    // cout << "Time cost: " << time(nullptr) - optimization_start_time << "s" << endl;
    // 对于file_path为fjsp.hurink.vdata-mt20.m5j20c5.txt 输出格式为：hurink.vdata-mt20.m5j20c5
    // cout << "search_time:" << time(nullptr) - optimization_start_time << "s" << endl;
    // cout << "repeat_count:" << repeat_count << endl;
    // cout << "count:" << count << endl;
    cout << "Final makespan: " << ans1 << endl;
    cout << ans1 << endl;
    cout << time(nullptr) - optimization_start_time << endl;
    cout << endl;
    return 0;
}
