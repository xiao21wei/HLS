#include <iostream>

#include "DataProc.h"
#include "AdaptiveSearch.h"
#include "GreedySearch.h"
#include "GreedyRlPolicy.h"
#include "GuidedSearch.h"
#include "RandomSearch.h"
#include "TabuSearch.h"
#include "TabuRlPolicy.h"
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
#include <limits>
#include <random>
#include <string.h>
#include <vector>

using namespace std;

namespace {

constexpr int kMaximumOuterCalls = 100000;
constexpr int kMaximumEliteArchiveSize = 8;

bool GuidedSearchRequested() {
    const char *value = std::getenv("HLS_ENABLE_GUIDED_SEARCH");
    return value == nullptr || std::string(value) != "0";
}

bool GuidedSearchForced() {
    const char *value = std::getenv("HLS_ENABLE_GUIDED_SEARCH");
    return value != nullptr && std::string(value) == "force";
}

int TargetMakespanFromEnvironment() {
    const char *value = std::getenv("HLS_TARGET_MAKESPAN");
    if (value == nullptr) {
        return 0;
    }
    try {
        return std::max(0, std::stoi(value));
    } catch (const std::exception &) {
        return 0;
    }
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

Schedule BuildDispatchInitialSolution(
    const std::vector<std::string> &job_list,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    std::mt19937 &random_engine
) {
    struct Slot { int start; int finish; Schedule_process process; };
    std::vector<const Job *> selected_jobs;
    for (const auto &name : job_list) {
        const auto found = std::find_if(jobs.begin(), jobs.end(), [&](const Job &job) {
            return job.get_job_name() == name;
        });
        if (found == jobs.end()) throw std::runtime_error("Dispatch initial solution: missing job");
        selected_jobs.push_back(&*found);
    }
    std::vector<int> next_process(selected_jobs.size(), 0), ready_at(selected_jobs.size(), 0);
    std::vector<double> remaining_work(selected_jobs.size(), 0.0), priority(selected_jobs.size());
    int remaining_operations = 0;
    for (int index = 0; index < static_cast<int>(selected_jobs.size()); ++index) {
        for (const auto &process : selected_jobs[index]->get_job_process()) {
            if (process.process_item.empty()) throw std::runtime_error("Dispatch initial solution: empty operation");
            int minimum = std::numeric_limits<int>::max();
            for (const auto &option : process.process_item) minimum = std::min(minimum, option.process_time);
            remaining_work[index] += minimum;
            ++remaining_operations;
        }
        priority[index] = std::uniform_real_distribution<double>(0.8, 1.2)(random_engine);
    }
    const double remaining_weight = std::uniform_real_distribution<double>(0.0, 1.2)(random_engine);
    const double duration_weight = std::uniform_real_distribution<double>(-0.1, 0.4)(random_engine);
    const double load_weight = std::uniform_real_distribution<double>(0.0, 0.25)(random_engine);
    std::vector<std::vector<Slot>> slots(machines.size());
    std::vector<int> workloads(machines.size(), 0);
    while (remaining_operations > 0) {
        double best_score = std::numeric_limits<double>::infinity();
        int best_job = -1, best_machine = -1, best_start = 0, best_duration = 0;
        for (int job = 0; job < static_cast<int>(selected_jobs.size()); ++job) {
            const auto &processes = selected_jobs[job]->get_job_process();
            if (next_process[job] >= static_cast<int>(processes.size())) continue;
            for (const auto &option : processes[next_process[job]].process_item) {
                const int machine = option.machine_id;
                if (machine < 0 || machine >= static_cast<int>(machines.size()) || option.process_time <= 0) {
                    throw std::runtime_error("Dispatch initial solution: invalid machine/duration");
                }
                int start = ready_at[job];
                for (const auto &slot : slots[machine]) {
                    if (start + option.process_time <= slot.start) break;
                    start = std::max(start, slot.finish);
                }
                const double score = start + option.process_time + duration_weight * option.process_time -
                    remaining_weight * priority[job] * remaining_work[job] + load_weight * workloads[machine];
                if (score < best_score) {
                    best_score = score;
                    best_job = job;
                    best_machine = machine;
                    best_start = start;
                    best_duration = option.process_time;
                }
            }
        }
        if (best_job < 0) throw std::runtime_error("Dispatch initial solution: no ready operation");
        const auto &process = selected_jobs[best_job]->get_job_process()[next_process[best_job]];
        Slot inserted{best_start, best_start + best_duration,
                      {selected_jobs[best_job]->get_job_id(), process.process_item.front().process_id}};
        auto &sequence = slots[best_machine];
        const auto position = std::lower_bound(sequence.begin(), sequence.end(), best_start,
            [](const Slot &slot, const int start) { return slot.start < start; });
        sequence.insert(position, inserted);
        workloads[best_machine] += best_duration;
        ready_at[best_job] = inserted.finish;
        int minimum = std::numeric_limits<int>::max();
        for (const auto &option : process.process_item) minimum = std::min(minimum, option.process_time);
        remaining_work[best_job] -= minimum;
        ++next_process[best_job];
        --remaining_operations;
    }
    Schedule result;
    result.set_machine_count(static_cast<int>(machines.size()));
    std::vector<Schedule_item> items(machines.size());
    for (int machine = 0; machine < static_cast<int>(machines.size()); ++machine) {
        items[machine].machine_id = machines[machine].get_machine_id();
        for (const auto &slot : slots[machine]) items[machine].schedule_process.push_back(slot.process);
        items[machine].process_count = static_cast<int>(items[machine].schedule_process.size());
    }
    result.set_schedule_items(items);
    ScheduleItemsToGraph(result, items, jobs, job_list, true, machines);
    int sparse = 0;
    if (!TryCalculateScheduleItemsMakespan(items, jobs, job_list, sparse) || sparse != result.get_TotalTime()) {
        throw std::runtime_error("Dispatch initial solution: feasibility or sparse/dense disagreement");
    }
    return result;
}

Schedule BuildMultiStartHeuristicInitialSolution(
    const std::vector<std::string> &job_list,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    const unsigned int random_seed,
    const SearchDeadline deadline,
    const int requested_starts = 0
) {
    Schedule best_schedule = GenerateInitialSolution(job_list, jobs, machines);
    if (best_schedule.get_graph().empty()) {
        return best_schedule;
    }

    // Randomized job orders expose different machine bottlenecks while the
    // EET constructor keeps every candidate precedence-feasible.
    const int start_count = requested_starts > 0
        ? requested_starts
        : std::min(32, std::max(8, static_cast<int>(jobs.size()) * 2));
    std::mt19937 random_engine(random_seed ^ 0xD1B54A32U);
    const char *dispatch_flag = std::getenv("HLS_DISPATCH_INITIAL");
    const bool dispatch_enabled = dispatch_flag != nullptr && std::string(dispatch_flag) == "1";
    std::vector<std::string> shuffled_job_list = job_list;
    for (int start = 1; start < start_count; ++start) {
        if (SearchDeadlineReached(deadline)) {
            break;
        }
        std::shuffle(shuffled_job_list.begin(), shuffled_job_list.end(), random_engine);
        Schedule candidate = dispatch_enabled && start % 2 == 1
            ? BuildDispatchInitialSolution(job_list, jobs, machines, random_engine)
            : GenerateInitialSolution(shuffled_job_list, jobs, machines);
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
    const int target_makespan = TargetMakespanFromEnvironment();
    ConfigureGreedyRlPolicyFromEnvironment(random_seed ^ 0x27D4EB2DU);
    const GreedyRlRuntimeConfig &greedy_rl_config =
        GetGreedyRlRuntimeConfig();
    ConfigureTabuRlPolicyFromEnvironment(random_seed ^ 0x165667B1U);
    const TabuRlRuntimeConfig &tabu_rl_config = GetTabuRlRuntimeConfig();

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
    cout << "target_makespan: " << target_makespan << endl;
    cout << "greedy_rl_requested: "
         << (greedy_rl_config.requested ? "true" : "false") << endl;
    cout << "greedy_rl_enabled: "
         << (greedy_rl_config.enabled ? "true" : "false") << endl;
    cout << "greedy_rl_model: " << greedy_rl_config.model_path << endl;
    cout << "greedy_rl_top_k: " << greedy_rl_config.candidate_limit << endl;
    cout << "greedy_rl_dynamic_top_k: "
         << (greedy_rl_config.dynamic_candidate_limit ? "true" : "false")
         << endl;
    cout << "greedy_rl_fallback_on_miss: "
         << (greedy_rl_config.fallback_on_miss ? "true" : "false")
         << endl;
    cout << "greedy_rl_verify_with_baseline: "
         << (greedy_rl_config.verify_with_baseline ? "true" : "false")
         << endl;
    cout << "greedy_rl_temperature: " << greedy_rl_config.temperature << endl;
    cout << "greedy_rl_exploration: "
         << greedy_rl_config.exploration_fraction << endl;
    cout << "greedy_rl_rule_fraction: " << greedy_rl_config.rule_fraction
         << endl;
    cout << "greedy_rl_status: " << greedy_rl_config.status << endl;
    cout << "tabu_rl_requested: "
         << (tabu_rl_config.requested ? "true" : "false") << endl;
    cout << "tabu_rl_enabled: "
         << (tabu_rl_config.enabled ? "true" : "false") << endl;
    cout << "tabu_rl_model: " << tabu_rl_config.model_path << endl;
    cout << "tabu_rl_top_k: " << tabu_rl_config.candidate_limit << endl;
    cout << "tabu_rl_dynamic_top_k: "
         << (tabu_rl_config.dynamic_candidate_limit ? "true" : "false")
         << endl;
    cout << "tabu_rl_fallback_on_miss: "
         << (tabu_rl_config.fallback_on_miss ? "true" : "false")
         << endl;
    cout << "tabu_rl_fallback_on_non_improvement: "
         << (tabu_rl_config.fallback_on_non_improvement ? "true" : "false")
         << endl;
    cout << "tabu_rl_verify_full_neighborhood: "
         << (tabu_rl_config.verify_full_neighborhood ? "true" : "false")
         << endl;
    cout << "tabu_rl_min_per_neighborhood: "
         << tabu_rl_config.minimum_per_neighborhood << endl;
    cout << "tabu_rl_temperature: " << tabu_rl_config.temperature << endl;
    cout << "tabu_rl_exploration: "
         << tabu_rl_config.exploration_fraction << endl;
    cout << "tabu_rl_rule_fraction: " << tabu_rl_config.rule_fraction << endl;
    cout << "tabu_rl_status: " << tabu_rl_config.status << endl;

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
    for (const SearchMethod method : {
             SearchMethod::Greedy,
             SearchMethod::Genetic,
             SearchMethod::Tabu,
             SearchMethod::Guided
         }) {
        cout << "method_score_bonus[" << SearchMethodName(method) << "]: "
             << search_controller.GetScoreBonus(method) << endl;
    }
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
    const char *restart_flag = std::getenv("HLS_ENABLE_RESTARTS");
    const bool restarts_enabled = restart_flag != nullptr &&
        std::string(restart_flag) == "1";
    int restart_stagnation_limit = 40;
    if (const char *value = std::getenv("HLS_RESTART_STAGNATION_CALLS")) {
        try {
            restart_stagnation_limit = std::clamp(std::stoi(value), 8, 1000);
        } catch (const std::exception &) {
            std::cerr << "Invalid HLS_RESTART_STAGNATION_CALLS; using 40." << endl;
        }
    }
    std::mt19937 restart_rng(random_seed ^ 0x94D049BBU);
    int restart_count = 0;
    int phase_stagnation_calls = 0;
    std::array<bool, 3> phase_methods_seen{};
    std::array<SearchMethodStats, 4> completed_phase_stats{};
    cout << "hybrid_restarts_enabled: "
         << (restarts_enabled ? "true" : "false") << endl;
    cout << "hybrid_restart_stagnation_calls: " << restart_stagnation_limit << endl;

    while (count <= kMaximumOuterCalls &&
           no_improvement_round_count <= max_repeat_count &&
           (target_makespan <= 0 ||
            output_schedule.get_TotalTime() > target_makespan) &&
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
        if (restarts_enabled && phase_stagnation_calls >= restart_stagnation_limit &&
            remaining_seconds >= 15.0 &&
            std::all_of(phase_methods_seen.begin(), phase_methods_seen.end(),
                        [](const bool seen) { return seen; })) {
            const unsigned int phase_seed = restart_rng();
            Schedule restart_schedule = BuildMultiStartHeuristicInitialSolution(
                jobList, jobs, machines, phase_seed,
                BuildCallDeadline(optimization_deadline, 2.0), 96
            );
            if (!restart_schedule.get_graph().empty()) {
                for (int method = 0; method < 4; ++method) {
                    const auto &stats = search_controller.GetStats(
                        static_cast<SearchMethod>(method));
                    auto &accumulated = completed_phase_stats[method];
                    accumulated.calls += stats.calls;
                    accumulated.improvements += stats.improvements;
                    accumulated.global_improvements += stats.global_improvements;
                    accumulated.total_relative_gain += stats.total_relative_gain;
                    accumulated.total_seconds += stats.total_seconds;
                }
                input_schedule = std::move(restart_schedule);
                if (input_schedule.get_TotalTime() < output_schedule.get_TotalTime()) {
                    output_schedule = input_schedule;
                }
                elite_archive.clear();
                AddEliteSchedule(elite_archive, output_schedule);
                AddEliteSchedule(elite_archive, input_schedule);
                ResetGeneticSearchState(phase_seed ^ 0x9E3779B9U);
                ResetTabuSearchState(phase_seed ^ 0x85EBCA6BU);
                ResetGuidedSearchState(phase_seed ^ 0xC2B2AE35U);
                ResetGreedySearchState();
                SetGreedyRlSamplingSeed(phase_seed ^ 0x27D4EB2DU);
                SetTabuRlSamplingSeed(phase_seed ^ 0x165667B1U);
                search_controller = AdaptiveSearchController(
                    search_mode_param, random_search_strategy_param,
                    phase_seed ^ 0xA511E9B3U, guided_search_enabled
                );
                phase_methods_seen.fill(false);
                phase_stagnation_calls = 0;
                stagnation_call_count = 0;
                method_failed_since_improvement.fill(false);
                method_failed_since_improvement[static_cast<int>(SearchMethod::Guided)] =
                    !guided_search_enabled;
                ++restart_count;
                cout << "Hybrid restart: phase=" << restart_count
                     << ", derived_seed=" << phase_seed
                     << ", initial_makespan=" << input_schedule.get_TotalTime()
                     << ", incumbent=" << output_schedule.get_TotalTime() << endl;
                continue;
            }
        }
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
        const int phase_before_makespan = input_schedule.get_TotalTime();
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
        if (static_cast<int>(decision.method) < 3) {
            phase_methods_seen[static_cast<int>(decision.method)] = true;
        }
        phase_stagnation_calls = local_after_makespan < phase_before_makespan
            ? 0 : phase_stagnation_calls + 1;
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
            if (restarts_enabled &&
                local_after_makespan < input_schedule.get_TotalTime()) {
                input_schedule = candidate_schedule;
                AddEliteSchedule(elite_archive, input_schedule);
            }
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
        SearchMethodStats stats = search_controller.GetStats(method);
        const auto &previous = completed_phase_stats[static_cast<int>(method)];
        stats.calls += previous.calls;
        stats.improvements += previous.improvements;
        stats.global_improvements += previous.global_improvements;
        stats.total_relative_gain += previous.total_relative_gain;
        stats.total_seconds += previous.total_seconds;
        cout << SearchMethodName(method)
             << " summary: calls=" << stats.calls
             << ", local_improvements=" << stats.improvements
             << ", global_improvements=" << stats.global_improvements
             << ", relative_gain=" << fixed << setprecision(6)
             << stats.total_relative_gain
             << ", time=" << setprecision(3) << stats.total_seconds
             << defaultfloat << "s" << endl;
    }
    if (greedy_rl_config.requested) {
        const GreedyRlStats &rl_stats = GetGreedyRlStats();
        cout << "Greedy RL summary: enabled="
             << (greedy_rl_config.enabled ? "true" : "false")
             << ", selection_calls=" << rl_stats.selection_calls
             << ", generated_candidates=" << rl_stats.generated_candidates
             << ", selected_candidates=" << rl_stats.selected_candidates
             << ", model_candidates=" << rl_stats.model_candidates
             << ", rule_candidates=" << rl_stats.rule_candidates
             << ", exploration_candidates="
             << rl_stats.exploration_candidates
             << ", fallback_calls=" << rl_stats.fallback_calls
             << ", verification_calls=" << rl_stats.verification_calls
             << endl;
    }
    if (tabu_rl_config.requested) {
        const TabuRlStats &rl_stats = GetTabuRlStats();
        cout << "Tabu RL summary: enabled="
             << (tabu_rl_config.enabled ? "true" : "false")
             << ", selection_calls=" << rl_stats.selection_calls
             << ", generated_candidates=" << rl_stats.generated_candidates
             << ", selected_candidates=" << rl_stats.selected_candidates
             << ", model_candidates=" << rl_stats.model_candidates
             << ", rule_candidates=" << rl_stats.rule_candidates
             << ", exploration_candidates="
             << rl_stats.exploration_candidates
             << ", neighborhood_safeguards="
             << rl_stats.neighborhood_safeguards
             << ", fallback_calls=" << rl_stats.fallback_calls
             << ", non_improvement_fallback_calls="
             << rl_stats.non_improvement_fallback_calls
             << ", intensification_expansion_calls="
             << rl_stats.intensification_expansion_calls
             << ", verification_calls=" << rl_stats.verification_calls
             << endl;
    }

    // output_schedule.to_string();
    // output_schedule.ToMapCode();
    const int ans1 = CalculateTotalTime(output_schedule);
    cout << "Hybrid restart summary: restarts=" << restart_count << endl;
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
