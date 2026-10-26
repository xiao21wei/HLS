//
// Created by luopw on 24-12-1.
//

#include "GreedySearch.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataProc.h"
#include "CoptInitialSolution.h"
#include "RouteRepairProfile.h"
#include "GreedyRlPolicy.h"

namespace {

constexpr int kMaxCriticalCandidates = 28;
constexpr int kMaxNearCriticalCandidates = 12;
constexpr int kMaxSequenceEvaluationsPerStep = 600;
constexpr int kMaxMachineMoveEvaluationsPerStep = 1800;
constexpr int kMaxNearCriticalSequenceEvaluations = 180;
constexpr int kMaxNearCriticalMachineEvaluations = 480;
constexpr int kMaxCompoundEvaluations = 240;
constexpr int kMaximumNeutralMoves = 4;

thread_local int gNextCandidateWindow = 0;
thread_local int gActiveCandidateWindow = 0;
thread_local int gRepairFailures = 0;

long long BuildProcessId(const int job_id, const int process_id) {
    return (static_cast<long long>(job_id) << 32) ^
           static_cast<unsigned int>(process_id);
}

std::string BuildProcessKey(const int job_id, const int process_id) {
    return std::to_string(job_id) + "-" + std::to_string(process_id);
}

std::optional<std::pair<int, int>> ParseProcessKey(std::string process_key) {
    if (process_key == "start" || process_key == "end") {
        return std::nullopt;
    }
    if (process_key.rfind("job", 0) == 0) {
        process_key.erase(0, 3);
    }

    const size_t split_position = process_key.find('-');
    if (split_position == std::string::npos) {
        return std::nullopt;
    }

    try {
        return std::make_pair(
            std::stoi(process_key.substr(0, split_position)),
            std::stoi(process_key.substr(split_position + 1))
        );
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

struct ProcessLocation {
    int machine_index;
    int item_index;
};

struct RankedProcess {
    std::string process_key;
    int slack;
    int processing_time;
    int alternative_gain;
    int machine_workload;
    int start_time;
};

enum class GreedyRlMoveType {
    Sequence = 0,
    Machine = 1,
    Swap = 2
};

struct GreedyRlAction {
    GreedyRlMoveType type = GreedyRlMoveType::Sequence;
    int source_machine_index = -1;
    int target_machine_index = -1;
    int source_position = -1;
    int target_position = -1;
    GreedyRlFeatures features{};
};

struct EvaluationBudget {
    int used = 0;
    int maximum = 0;
    SearchDeadline deadline = SearchDeadline::max();

    bool Take() {
        if (used >= maximum || SearchDeadlineReached(deadline)) {
            return false;
        }
        ++used;
        return true;
    }

    bool Exhausted() const {
        return used >= maximum || SearchDeadlineReached(deadline);
    }
};

struct ScheduleQuality {
    int makespan;
    int critical_process_count;
    long long total_slack;
};

ScheduleQuality BuildScheduleQuality(const Schedule &schedule) {
    ScheduleQuality quality{
        schedule.get_TotalTime(),
        0,
        0
    };
    const auto process_slack = schedule.GetProcessSlack();
    for (const auto &[process_key, slack] : process_slack) {
        if (process_key == "start" || process_key == "end") {
            continue;
        }
        if (slack == 0) {
            ++quality.critical_process_count;
        }
        quality.total_slack += slack;
    }
    return quality;
}

bool IsBetterSchedule(
    const Schedule &candidate,
    const Schedule &current
) {
    if (candidate.get_TotalTime() != current.get_TotalTime()) {
        return candidate.get_TotalTime() < current.get_TotalTime();
    }
    const ScheduleQuality candidate_quality = BuildScheduleQuality(candidate);
    const ScheduleQuality current_quality = BuildScheduleQuality(current);
    return std::make_tuple(
               candidate_quality.makespan,
               candidate_quality.critical_process_count,
               -candidate_quality.total_slack
           ) < std::make_tuple(
               current_quality.makespan,
               current_quality.critical_process_count,
               -current_quality.total_slack
           );
}

std::unordered_map<long long, ProcessLocation> BuildProcessLocationMap(
    const std::vector<Schedule_item> &schedule_items
) {
    std::unordered_map<long long, ProcessLocation> location_map;
    int process_count = 0;
    for (const auto &schedule_item : schedule_items) {
        process_count += schedule_item.process_count;
    }
    location_map.reserve(process_count);

    for (int machine_index = 0; machine_index < static_cast<int>(schedule_items.size()); ++machine_index) {
        const auto &processes = schedule_items[machine_index].schedule_process;
        for (int item_index = 0; item_index < static_cast<int>(processes.size()); ++item_index) {
            const auto &process = processes[item_index];
            location_map[BuildProcessId(process.job_id, process.process_id)] = {
                machine_index,
                item_index
            };
        }
    }
    return location_map;
}

std::unordered_map<int, int> BuildMachineIndexMap(
    const std::vector<Schedule_item> &schedule_items
) {
    std::unordered_map<int, int> machine_index_map;
    machine_index_map.reserve(schedule_items.size());
    for (int index = 0; index < static_cast<int>(schedule_items.size()); ++index) {
        machine_index_map[schedule_items[index].machine_id] = index;
    }
    return machine_index_map;
}

std::unordered_map<int, const Job *> BuildJobIdMap(const std::vector<Job> &jobs) {
    std::unordered_map<int, const Job *> job_id_map;
    job_id_map.reserve(jobs.size());
    for (const auto &job : jobs) {
        job_id_map[job.get_job_id()] = &job;
    }
    return job_id_map;
}

const Job_process *FindJobProcess(const Job &job, const int process_id) {
    for (const auto &job_process : job.get_job_process()) {
        if (!job_process.process_item.empty() &&
            job_process.process_item.front().process_id == process_id) {
            return &job_process;
        }
    }
    return nullptr;
}

int GetProcessingTime(const Job_process &job_process, const int machine_id) {
    for (const auto &process_item : job_process.process_item) {
        if (process_item.machine_id == machine_id) {
            return process_item.process_time;
        }
    }
    return 0;
}

std::vector<std::string> BuildProcessListFromScheduleItems(
    const std::vector<Schedule_item> &schedule_items
) {
    int process_count = 2;
    for (const auto &schedule_item : schedule_items) {
        process_count += schedule_item.process_count;
    }

    std::vector<std::string> process_list;
    process_list.reserve(process_count);
    process_list.push_back("start");
    for (const auto &schedule_item : schedule_items) {
        for (const auto &process : schedule_item.schedule_process) {
            process_list.push_back(BuildProcessKey(process.job_id, process.process_id));
        }
    }
    process_list.push_back("end");
    return process_list;
}

std::vector<int> BuildMachineWorkloads(
    const std::vector<Schedule_item> &schedule_items,
    const std::unordered_map<int, const Job *> &job_id_map
) {
    std::vector<int> workloads(schedule_items.size(), 0);
    for (int machine_index = 0; machine_index < static_cast<int>(schedule_items.size()); ++machine_index) {
        const auto &schedule_item = schedule_items[machine_index];
        for (const auto &process : schedule_item.schedule_process) {
            const auto job_it = job_id_map.find(process.job_id);
            if (job_it == job_id_map.end()) {
                continue;
            }
            const Job_process *job_process = FindJobProcess(*job_it->second, process.process_id);
            if (job_process != nullptr) {
                workloads[machine_index] += GetProcessingTime(*job_process, schedule_item.machine_id);
            }
        }
    }
    return workloads;
}

std::unordered_map<long long, int> BuildStartTimeMap(const Schedule &schedule) {
    std::unordered_map<long long, int> start_time_map;
    const auto &process_list = schedule.get_processList();
    const auto &start_times = schedule.get_start_time();
    const int count = std::min(process_list.size(), start_times.size());
    start_time_map.reserve(count);

    for (int index = 0; index < count; ++index) {
        const auto parsed_process = ParseProcessKey(process_list[index]);
        if (parsed_process.has_value()) {
            start_time_map[BuildProcessId(parsed_process->first, parsed_process->second)] =
                start_times[index];
        }
    }
    return start_time_map;
}

int CountOperations(const std::vector<Schedule_item> &schedule_items) {
    int operation_count = 0;
    for (const auto &schedule_item : schedule_items) {
        operation_count += static_cast<int>(schedule_item.schedule_process.size());
    }
    return operation_count;
}

double BoundedRatio(const double numerator, const double denominator) {
    return std::clamp(
        numerator / std::max(1.0, denominator),
        -2.0,
        2.0
    );
}

GreedyRlFeatures BuildGreedyRlFeatures(
    const GreedyRlMoveType move_type,
    const int slack,
    const int current_processing_time,
    const int target_processing_time,
    const int source_workload,
    const int target_workload,
    const int source_position,
    const int target_position,
    const int source_length,
    const int target_length,
    const int operation_count,
    const int makespan
) {
    GreedyRlFeatures features{};
    features[0] = move_type == GreedyRlMoveType::Sequence ? 1.0 : 0.0;
    features[1] = move_type == GreedyRlMoveType::Machine ? 1.0 : 0.0;
    features[2] = move_type == GreedyRlMoveType::Swap ? 1.0 : 0.0;
    features[3] = slack == 0 ? 1.0 : 0.0;
    features[4] = BoundedRatio(slack, makespan);
    features[5] = BoundedRatio(current_processing_time, makespan);
    features[6] = BoundedRatio(target_processing_time, makespan);
    features[7] = BoundedRatio(
        current_processing_time - target_processing_time,
        makespan
    );
    features[8] = BoundedRatio(source_workload, makespan);
    features[9] = BoundedRatio(target_workload, makespan);
    features[10] = BoundedRatio(
        source_workload - target_workload,
        makespan
    );
    features[11] = BoundedRatio(source_position, source_length - 1);
    features[12] = BoundedRatio(target_position, target_length);
    features[13] = BoundedRatio(
        std::abs(target_position - source_position),
        std::max(source_length, target_length)
    );
    features[14] = BoundedRatio(source_length, operation_count);
    features[15] = BoundedRatio(target_length, operation_count);
    return features;
}

void GenerateRlSequenceActions(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &process_list,
    std::vector<GreedyRlAction> &actions
) {
    const auto &items = schedule.get_schedule_items();
    const auto location_map = BuildProcessLocationMap(items);
    const auto job_id_map = BuildJobIdMap(jobs);
    const auto workloads = BuildMachineWorkloads(items, job_id_map);
    const auto slack = schedule.GetProcessSlack();
    const int operation_count = CountOperations(items);

    for (const auto &process_key : process_list) {
        const auto process = ParseProcessKey(process_key);
        if (!process.has_value()) {
            continue;
        }
        const auto location_it = location_map.find(BuildProcessId(
            process->first,
            process->second
        ));
        const auto job_it = job_id_map.find(process->first);
        if (location_it == location_map.end() || job_it == job_id_map.end()) {
            continue;
        }
        const Job_process *job_process = FindJobProcess(
            *job_it->second,
            process->second
        );
        if (job_process == nullptr) {
            continue;
        }
        const int machine_index = location_it->second.machine_index;
        const int source_position = location_it->second.item_index;
        const int process_count = static_cast<int>(
            items[machine_index].schedule_process.size()
        );
        if (process_count <= 1) {
            continue;
        }
        const int processing_time = GetProcessingTime(
            *job_process,
            items[machine_index].machine_id
        );
        const int process_slack = slack.contains(process_key)
            ? slack.at(process_key)
            : 0;
        for (int target_position = 0;
             target_position < process_count;
             ++target_position) {
            if (target_position == source_position) {
                continue;
            }
            actions.push_back({
                GreedyRlMoveType::Sequence,
                machine_index,
                machine_index,
                source_position,
                target_position,
                BuildGreedyRlFeatures(
                    GreedyRlMoveType::Sequence,
                    process_slack,
                    processing_time,
                    processing_time,
                    workloads[machine_index],
                    workloads[machine_index],
                    source_position,
                    target_position,
                    process_count,
                    process_count - 1,
                    operation_count,
                    schedule.get_TotalTime()
                )
            });
        }
    }
}

void GenerateRlMachineActions(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &process_list,
    std::vector<GreedyRlAction> &actions
) {
    const auto &items = schedule.get_schedule_items();
    const auto location_map = BuildProcessLocationMap(items);
    const auto machine_index_map = BuildMachineIndexMap(items);
    const auto job_id_map = BuildJobIdMap(jobs);
    const auto workloads = BuildMachineWorkloads(items, job_id_map);
    const auto slack = schedule.GetProcessSlack();
    const int operation_count = CountOperations(items);

    for (const auto &process_key : process_list) {
        const auto process = ParseProcessKey(process_key);
        if (!process.has_value()) {
            continue;
        }
        const auto location_it = location_map.find(BuildProcessId(
            process->first,
            process->second
        ));
        const auto job_it = job_id_map.find(process->first);
        if (location_it == location_map.end() || job_it == job_id_map.end()) {
            continue;
        }
        const Job_process *job_process = FindJobProcess(
            *job_it->second,
            process->second
        );
        if (job_process == nullptr || job_process->process_item.size() <= 1) {
            continue;
        }

        const int source_machine_index = location_it->second.machine_index;
        const int source_position = location_it->second.item_index;
        const int source_machine_id = items[source_machine_index].machine_id;
        const int source_length = static_cast<int>(
            items[source_machine_index].schedule_process.size()
        );
        const int current_processing_time = GetProcessingTime(
            *job_process,
            source_machine_id
        );
        const int process_slack = slack.contains(process_key)
            ? slack.at(process_key)
            : 0;

        std::vector<const Process_item *> machine_options;
        for (const auto &option : job_process->process_item) {
            machine_options.push_back(&option);
        }
        std::stable_sort(
            machine_options.begin(),
            machine_options.end(),
            [&machine_index_map, &workloads](
                const Process_item *left,
                const Process_item *right
            ) {
                const auto left_it = machine_index_map.find(left->machine_id);
                const auto right_it = machine_index_map.find(right->machine_id);
                const int left_estimate = left_it == machine_index_map.end()
                    ? INT_MAX
                    : workloads[left_it->second] + left->process_time;
                const int right_estimate = right_it == machine_index_map.end()
                    ? INT_MAX
                    : workloads[right_it->second] + right->process_time;
                if (left_estimate != right_estimate) {
                    return left_estimate < right_estimate;
                }
                return left->process_time < right->process_time;
            }
        );

        std::unordered_set<int> visited_machines;
        for (const Process_item *option : machine_options) {
            if (option->machine_id == source_machine_id ||
                !visited_machines.insert(option->machine_id).second) {
                continue;
            }
            const auto target_it = machine_index_map.find(option->machine_id);
            if (target_it == machine_index_map.end()) {
                continue;
            }
            const int target_machine_index = target_it->second;
            const int target_length = static_cast<int>(
                items[target_machine_index].schedule_process.size()
            );
            for (int target_position = 0;
                 target_position <= target_length;
                 ++target_position) {
                actions.push_back({
                    GreedyRlMoveType::Machine,
                    source_machine_index,
                    target_machine_index,
                    source_position,
                    target_position,
                    BuildGreedyRlFeatures(
                        GreedyRlMoveType::Machine,
                        process_slack,
                        current_processing_time,
                        option->process_time,
                        workloads[source_machine_index],
                        workloads[target_machine_index],
                        source_position,
                        target_position,
                        source_length,
                        target_length,
                        operation_count,
                        schedule.get_TotalTime()
                    )
                });
            }
        }
    }
}

void GenerateRlSwapActions(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &process_list,
    std::vector<GreedyRlAction> &actions
) {
    const auto &items = schedule.get_schedule_items();
    const auto location_map = BuildProcessLocationMap(items);
    const auto job_id_map = BuildJobIdMap(jobs);
    const auto workloads = BuildMachineWorkloads(items, job_id_map);
    const auto slack = schedule.GetProcessSlack();
    const int operation_count = CountOperations(items);
    struct SwapProcess {
        int machine_index;
        int position;
        int processing_time;
        int slack;
    };
    std::vector<SwapProcess> processes;
    for (const auto &process_key : process_list) {
        const auto process = ParseProcessKey(process_key);
        if (!process.has_value()) {
            continue;
        }
        const auto location_it = location_map.find(BuildProcessId(
            process->first,
            process->second
        ));
        const auto job_it = job_id_map.find(process->first);
        if (location_it == location_map.end() || job_it == job_id_map.end()) {
            continue;
        }
        const Job_process *job_process = FindJobProcess(
            *job_it->second,
            process->second
        );
        if (job_process == nullptr) {
            continue;
        }
        processes.push_back({
            location_it->second.machine_index,
            location_it->second.item_index,
            GetProcessingTime(
                *job_process,
                items[location_it->second.machine_index].machine_id
            ),
            slack.contains(process_key) ? slack.at(process_key) : 0
        });
    }

    for (int left = 0; left < static_cast<int>(processes.size()); ++left) {
        for (int right = left + 1;
             right < static_cast<int>(processes.size());
             ++right) {
            if (processes[left].machine_index != processes[right].machine_index ||
                processes[left].position == processes[right].position) {
                continue;
            }
            const int machine_index = processes[left].machine_index;
            const int process_count = static_cast<int>(
                items[machine_index].schedule_process.size()
            );
            actions.push_back({
                GreedyRlMoveType::Swap,
                machine_index,
                machine_index,
                processes[left].position,
                processes[right].position,
                BuildGreedyRlFeatures(
                    GreedyRlMoveType::Swap,
                    processes[left].slack,
                    processes[left].processing_time,
                    processes[right].processing_time,
                    workloads[machine_index],
                    workloads[machine_index],
                    processes[left].position,
                    processes[right].position,
                    process_count,
                    process_count,
                    operation_count,
                    schedule.get_TotalTime()
                )
            });
        }
    }
}

std::vector<Schedule_item> ApplyGreedyRlAction(
    const std::vector<Schedule_item> &original_items,
    const GreedyRlAction &action
) {
    std::vector<Schedule_item> candidate_items = original_items;
    if (action.type == GreedyRlMoveType::Swap) {
        auto &processes =
            candidate_items[action.source_machine_index].schedule_process;
        std::swap(
            processes[action.source_position],
            processes[action.target_position]
        );
        return candidate_items;
    }

    auto &source_processes =
        candidate_items[action.source_machine_index].schedule_process;
    const Schedule_process moved_process = source_processes[action.source_position];
    source_processes.erase(source_processes.begin() + action.source_position);
    candidate_items[action.source_machine_index].process_count =
        static_cast<int>(source_processes.size());
    auto &target_processes =
        candidate_items[action.target_machine_index].schedule_process;
    target_processes.insert(
        target_processes.begin() + action.target_position,
        moved_process
    );
    candidate_items[action.target_machine_index].process_count =
        static_cast<int>(target_processes.size());
    return candidate_items;
}

std::vector<std::string> RankCriticalProcesses(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const bool flexible_only,
    const bool near_critical_only = false
) {
    const auto schedule_items = schedule.get_schedule_items();
    const auto location_map = BuildProcessLocationMap(schedule_items);
    const auto job_id_map = BuildJobIdMap(jobs);
    const auto workloads = BuildMachineWorkloads(schedule_items, job_id_map);
    const auto start_time_map = BuildStartTimeMap(schedule);
    const auto process_slack = schedule.GetProcessSlack();
    const int near_critical_limit = std::max(
        6,
        std::min(20, schedule.get_TotalTime() / 40)
    );

    std::vector<RankedProcess> ranked_processes;
    std::unordered_set<long long> seen_processes;
    for (const auto &process_key : schedule.get_processList()) {
        const auto parsed_process = ParseProcessKey(process_key);
        if (!parsed_process.has_value()) {
            continue;
        }

        const int job_id = parsed_process->first;
        const int process_id = parsed_process->second;
        const long long process_identifier = BuildProcessId(job_id, process_id);
        if (!seen_processes.insert(process_identifier).second) {
            continue;
        }

        const auto location_it = location_map.find(process_identifier);
        const auto job_it = job_id_map.find(job_id);
        if (location_it == location_map.end() || job_it == job_id_map.end()) {
            continue;
        }
        const auto slack_it = process_slack.find(process_key);
        if (slack_it == process_slack.end() ||
            (!near_critical_only && slack_it->second != 0) ||
            (near_critical_only &&
             (slack_it->second == 0 ||
              slack_it->second > near_critical_limit))) {
            continue;
        }

        const Job_process *job_process = FindJobProcess(*job_it->second, process_id);
        if (job_process == nullptr || job_process->process_item.empty()) {
            continue;
        }
        if (flexible_only && job_process->process_item.size() <= 1) {
            continue;
        }

        const int machine_index = location_it->second.machine_index;
        const int machine_id = schedule_items[machine_index].machine_id;
        const int current_processing_time = GetProcessingTime(*job_process, machine_id);
        int minimum_processing_time = INT_MAX;
        for (const auto &option : job_process->process_item) {
            minimum_processing_time = std::min(minimum_processing_time, option.process_time);
        }

        const auto start_time_it = start_time_map.find(process_identifier);
        ranked_processes.push_back({
            BuildProcessKey(job_id, process_id),
            slack_it->second,
            current_processing_time,
            std::max(0, current_processing_time - minimum_processing_time),
            workloads[machine_index],
            start_time_it == start_time_map.end() ? 0 : start_time_it->second
        });
    }

    std::sort(
        ranked_processes.begin(),
        ranked_processes.end(),
        [flexible_only](const RankedProcess &left, const RankedProcess &right) {
            if (left.slack != right.slack) {
                return left.slack < right.slack;
            }
            if (flexible_only && left.alternative_gain != right.alternative_gain) {
                return left.alternative_gain > right.alternative_gain;
            }
            if (left.machine_workload != right.machine_workload) {
                return left.machine_workload > right.machine_workload;
            }
            if (left.processing_time != right.processing_time) {
                return left.processing_time > right.processing_time;
            }
            return left.start_time > right.start_time;
        }
    );

    const int candidate_limit = near_critical_only
        ? kMaxNearCriticalCandidates
        : kMaxCriticalCandidates;
    if (static_cast<int>(ranked_processes.size()) > candidate_limit) {
        const int fixed_prefix = std::min(
            near_critical_only ? 3 : 8,
            candidate_limit
        );
        std::vector<RankedProcess> selected;
        selected.reserve(candidate_limit);
        selected.insert(
            selected.end(),
            ranked_processes.begin(),
            ranked_processes.begin() + fixed_prefix
        );
        const int rotating_count = candidate_limit - fixed_prefix;
        const int tail_count =
            static_cast<int>(ranked_processes.size()) - fixed_prefix;
        const int offset = tail_count == 0
            ? 0
            : (gActiveCandidateWindow * std::max(1, rotating_count)) %
                tail_count;
        for (int index = 0; index < rotating_count; ++index) {
            selected.push_back(ranked_processes[
                fixed_prefix + (offset + index) % tail_count
            ]);
        }
        ranked_processes = std::move(selected);
    }

    std::vector<std::string> result;
    result.reserve(ranked_processes.size());
    for (const auto &ranked_process : ranked_processes) {
        result.push_back(ranked_process.process_key);
    }
    return result;
}

std::vector<std::string> ExpandRepairProcessesWithJobNeighbors(
    const std::vector<std::string> &ranked_processes,
    const std::vector<Job> &jobs
) {
    const auto job_id_map = BuildJobIdMap(jobs);
    std::vector<std::string> expanded;
    std::unordered_set<std::string> seen;
    const int anchor_count = std::min(
        6,
        static_cast<int>(ranked_processes.size())
    );
    expanded.reserve(ranked_processes.size() + anchor_count * 4);
    for (int anchor = 0; anchor < anchor_count; ++anchor) {
        const auto parsed = ParseProcessKey(ranked_processes[anchor]);
        if (!parsed.has_value() || !job_id_map.contains(parsed->first)) {
            continue;
        }
        const auto &processes = job_id_map.at(parsed->first)->get_job_process();
        int process_index = -1;
        for (int index = 0; index < static_cast<int>(processes.size()); ++index) {
            if (!processes[index].process_item.empty() &&
                processes[index].process_item.front().process_id ==
                    parsed->second) {
                process_index = index;
                break;
            }
        }
        auto append = [&](const int index) {
            if (index < 0 || index >= static_cast<int>(processes.size()) ||
                processes[index].process_item.size() <= 1) {
                return;
            }
            const std::string key = BuildProcessKey(
                parsed->first,
                processes[index].process_item.front().process_id
            );
            if (seen.insert(key).second) {
                expanded.push_back(key);
            }
        };
        append(process_index);
        append(process_index - 1);
        append(process_index + 1);
        append(process_index - 2);
        append(process_index + 2);
    }
    for (const auto &key : ranked_processes) {
        if (seen.insert(key).second) {
            expanded.push_back(key);
        }
    }
    return expanded;
}

bool EvaluateCandidate(
    const Schedule &base_schedule,
    std::vector<Schedule_item> schedule_items,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    std::vector<Machine> &machines,
    Schedule &candidate,
    const int maximum_makespan
) {
    int sparse_makespan = 0;
    if (!TryCalculateScheduleItemsMakespan(
            schedule_items,
            jobs,
            job_list,
            sparse_makespan
        ) ||
        sparse_makespan > maximum_makespan) {
        return false;
    }
    candidate = base_schedule;
    candidate.set_schedule_id(base_schedule.get_schedule_id() + 1);
    candidate.set_schedule_items(schedule_items);

    const auto graph = ScheduleItemsToGraph(
        candidate,
        schedule_items,
        jobs,
        job_list,
        false,
        machines
    );
    candidate.set_graph(graph);
    candidate.set_processList(BuildProcessListFromScheduleItems(schedule_items));

    try {
        CalculateTotalTime(candidate);
    } catch (const std::runtime_error &) {
        return false;
    }
    if (candidate.get_TotalTime() != sparse_makespan) {
        throw std::runtime_error(
            "sparse Greedy evaluator disagrees with dense evaluator"
        );
    }
    return true;
}

void ConsiderCandidate(
    const Schedule &base_schedule,
    std::vector<Schedule_item> schedule_items,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    std::vector<Machine> &machines,
    EvaluationBudget &budget,
    Schedule &best_schedule
) {
    if (!budget.Take()) {
        return;
    }

    Schedule candidate;
    if (!EvaluateCandidate(
            base_schedule,
            std::move(schedule_items),
            jobs,
            job_list,
            machines,
            candidate,
            best_schedule.get_TotalTime()
        )) {
        return;
    }

    if (IsBetterSchedule(candidate, best_schedule)) {
        CalculateTotalFailureRate(candidate, machines);
        best_schedule = std::move(candidate);
    }
}

Schedule FindBestRlNeighbor(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    const std::vector<std::string> &sequence_processes,
    const std::vector<std::string> &machine_processes,
    std::vector<Machine> &machines,
    const bool include_sequence,
    const bool include_machine,
    const bool include_swap,
    const int maximum_evaluations,
    const SearchDeadline deadline
) {
    std::vector<GreedyRlAction> actions;
    if (include_sequence) {
        GenerateRlSequenceActions(schedule, jobs, sequence_processes, actions);
    }
    if (include_machine) {
        GenerateRlMachineActions(schedule, jobs, machine_processes, actions);
    }
    if (include_swap) {
        GenerateRlSwapActions(schedule, jobs, sequence_processes, actions);
    }
    if (actions.empty()) {
        return schedule;
    }

    std::vector<GreedyRlFeatures> features;
    features.reserve(actions.size());
    for (const auto &action : actions) {
        features.push_back(action.features);
    }
    const std::vector<std::size_t> selected_indices =
        SelectGreedyRlCandidates(
            features,
            maximum_evaluations,
            CountOperations(schedule.get_schedule_items())
        );
    EvaluationBudget budget{0, maximum_evaluations, deadline};
    Schedule best_schedule = schedule;
    const auto &original_items = schedule.get_schedule_items();
    for (const std::size_t index : selected_indices) {
        if (index >= actions.size() || !budget.Take()) {
            break;
        }
        Schedule candidate;
        if (!EvaluateCandidate(
                schedule,
                ApplyGreedyRlAction(original_items, actions[index]),
                jobs,
                job_list,
                machines,
                candidate,
                best_schedule.get_TotalTime()
            )) {
            continue;
        }
        if (IsBetterSchedule(candidate, best_schedule)) {
            CalculateTotalFailureRate(candidate, machines);
            best_schedule = std::move(candidate);
        }
    }
    return best_schedule;
}

Schedule FindBestSequenceNeighbor(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    const std::vector<std::string> &process_list,
    std::vector<Machine> &machines,
    EvaluationBudget &budget
) {
    Schedule best_schedule = schedule;
    const auto original_items = schedule.get_schedule_items();
    const auto location_map = BuildProcessLocationMap(original_items);

    for (const auto &process_key : process_list) {
        const auto parsed_process = ParseProcessKey(process_key);
        if (!parsed_process.has_value()) {
            continue;
        }
        const auto location_it = location_map.find(
            BuildProcessId(parsed_process->first, parsed_process->second)
        );
        if (location_it == location_map.end()) {
            continue;
        }

        const int machine_index = location_it->second.machine_index;
        const int original_position = location_it->second.item_index;
        const int machine_process_count = original_items[machine_index].process_count;
        if (machine_process_count <= 1) {
            continue;
        }

        const Schedule_process moved_process =
            original_items[machine_index].schedule_process[original_position];
        for (int insertion_position = 0;
             insertion_position < machine_process_count;
             ++insertion_position) {
            if (insertion_position == original_position) {
                continue;
            }

            std::vector<Schedule_item> candidate_items = original_items;
            auto &machine_processes = candidate_items[machine_index].schedule_process;
            machine_processes.erase(machine_processes.begin() + original_position);
            machine_processes.insert(
                machine_processes.begin() + insertion_position,
                moved_process
            );
            ConsiderCandidate(
                schedule,
                std::move(candidate_items),
                jobs,
                job_list,
                machines,
                budget,
                best_schedule
            );
            if (budget.Exhausted()) {
                return best_schedule;
            }
        }
    }
    return best_schedule;
}

Schedule FindBestMachineMoveNeighbor(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    const std::vector<std::string> &process_list,
    std::vector<Machine> &machines,
    EvaluationBudget &budget
) {
    Schedule best_schedule = schedule;
    const auto original_items = schedule.get_schedule_items();
    const auto location_map = BuildProcessLocationMap(original_items);
    const auto machine_index_map = BuildMachineIndexMap(original_items);
    const auto job_id_map = BuildJobIdMap(jobs);
    const auto machine_workloads = BuildMachineWorkloads(
        original_items,
        job_id_map
    );

    for (const auto &process_key : process_list) {
        const auto parsed_process = ParseProcessKey(process_key);
        if (!parsed_process.has_value()) {
            continue;
        }

        const int job_id = parsed_process->first;
        const int process_id = parsed_process->second;
        const auto location_it = location_map.find(BuildProcessId(job_id, process_id));
        const auto job_it = job_id_map.find(job_id);
        if (location_it == location_map.end() || job_it == job_id_map.end()) {
            continue;
        }

        const Job_process *job_process = FindJobProcess(*job_it->second, process_id);
        if (job_process == nullptr || job_process->process_item.size() <= 1) {
            continue;
        }

        const int source_machine_index = location_it->second.machine_index;
        const int source_position = location_it->second.item_index;
        const int source_machine_id = original_items[source_machine_index].machine_id;
        const Schedule_process moved_process =
            original_items[source_machine_index].schedule_process[source_position];
        std::unordered_set<int> visited_machines;

        std::vector<const Process_item *> machine_options;
        machine_options.reserve(job_process->process_item.size());
        for (const auto &machine_option : job_process->process_item) {
            machine_options.push_back(&machine_option);
        }
        std::stable_sort(
            machine_options.begin(),
            machine_options.end(),
            [&machine_index_map, &machine_workloads](
                const Process_item *left,
                const Process_item *right
            ) {
                const auto left_index = machine_index_map.find(left->machine_id);
                const auto right_index = machine_index_map.find(right->machine_id);
                const int left_workload = left_index == machine_index_map.end()
                    ? INT_MAX
                    : machine_workloads[left_index->second];
                const int right_workload = right_index == machine_index_map.end()
                    ? INT_MAX
                    : machine_workloads[right_index->second];
                const int left_estimate = left_workload + left->process_time;
                const int right_estimate = right_workload + right->process_time;
                if (left_estimate != right_estimate) {
                    return left_estimate < right_estimate;
                }
                return left->process_time < right->process_time;
            }
        );

        for (const Process_item *machine_option : machine_options) {
            const int target_machine_id = machine_option->machine_id;
            if (target_machine_id == source_machine_id ||
                !visited_machines.insert(target_machine_id).second) {
                continue;
            }

            const auto target_machine_it = machine_index_map.find(target_machine_id);
            if (target_machine_it == machine_index_map.end()) {
                continue;
            }
            const int target_machine_index = target_machine_it->second;
            const int target_process_count = original_items[target_machine_index].process_count;

            for (int insertion_position = 0;
                 insertion_position <= target_process_count;
                 ++insertion_position) {
                std::vector<Schedule_item> candidate_items = original_items;
                auto &source_processes = candidate_items[source_machine_index].schedule_process;
                source_processes.erase(source_processes.begin() + source_position);
                candidate_items[source_machine_index].process_count =
                    static_cast<int>(source_processes.size());

                auto &target_processes = candidate_items[target_machine_index].schedule_process;
                target_processes.insert(
                    target_processes.begin() + insertion_position,
                    moved_process
                );
                candidate_items[target_machine_index].process_count =
                    static_cast<int>(target_processes.size());

                ConsiderCandidate(
                    schedule,
                    std::move(candidate_items),
                    jobs,
                    job_list,
                    machines,
                    budget,
                    best_schedule
                );
                if (budget.Exhausted()) {
                    return best_schedule;
                }
            }
        }
    }
    return best_schedule;
}

Schedule FindBestCompoundNeighbor(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    const std::vector<std::string> &process_list,
    std::vector<Machine> &machines,
    EvaluationBudget &budget
) {
    Schedule best_schedule = schedule;
    const auto original_items = schedule.get_schedule_items();
    const auto location_map = BuildProcessLocationMap(original_items);
    std::vector<std::pair<int, int>> locations;
    locations.reserve(process_list.size());
    for (const auto &process_key : process_list) {
        const auto parsed_process = ParseProcessKey(process_key);
        if (!parsed_process.has_value()) {
            continue;
        }
        const auto location_it = location_map.find(
            BuildProcessId(parsed_process->first, parsed_process->second)
        );
        if (location_it != location_map.end()) {
            locations.emplace_back(
                location_it->second.machine_index,
                location_it->second.item_index
            );
        }
    }

    for (int left = 0; left < static_cast<int>(locations.size()); ++left) {
        for (int right = left + 1;
             right < static_cast<int>(locations.size());
             ++right) {
            if (!budget.Take()) {
                return best_schedule;
            }
            if (locations[left].first != locations[right].first ||
                locations[left].second == locations[right].second) {
                continue;
            }
            std::vector<Schedule_item> candidate_items = original_items;
            auto &processes = candidate_items[locations[left].first].schedule_process;
            std::swap(
                processes[locations[left].second],
                processes[locations[right].second]
            );
            Schedule candidate;
            if (!EvaluateCandidate(
                    schedule,
                    std::move(candidate_items),
                    jobs,
                    job_list,
                    machines,
                    candidate,
                    best_schedule.get_TotalTime()
                )) {
                continue;
            }
            if (IsBetterSchedule(candidate, best_schedule)) {
                CalculateTotalFailureRate(candidate, machines);
                best_schedule = std::move(candidate);
            }
        }
    }
    return best_schedule;
}

Schedule FindBestBaselineGreedyStep(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    std::vector<Machine> &machines,
    const SearchDeadline deadline
) {
    const auto sequence_processes = RankCriticalProcesses(
        schedule,
        jobs,
        false
    );
    const auto move_processes = RankCriticalProcesses(
        schedule,
        jobs,
        true
    );
    EvaluationBudget sequence_budget{
        0,
        kMaxSequenceEvaluationsPerStep,
        deadline
    };
    EvaluationBudget move_budget{
        0,
        kMaxMachineMoveEvaluationsPerStep,
        deadline
    };
    Schedule best_schedule = FindBestSequenceNeighbor(
        schedule,
        jobs,
        job_list,
        sequence_processes,
        machines,
        sequence_budget
    );
    const Schedule move_schedule = FindBestMachineMoveNeighbor(
        schedule,
        jobs,
        job_list,
        move_processes,
        machines,
        move_budget
    );
    if (IsBetterSchedule(move_schedule, best_schedule)) {
        best_schedule = move_schedule;
    }

    if (best_schedule.get_TotalTime() >= schedule.get_TotalTime() &&
        !SearchDeadlineReached(deadline)) {
        const auto near_sequence_processes = RankCriticalProcesses(
            schedule,
            jobs,
            false,
            true
        );
        const auto near_move_processes = RankCriticalProcesses(
            schedule,
            jobs,
            true,
            true
        );
        EvaluationBudget near_sequence_budget{
            0,
            kMaxNearCriticalSequenceEvaluations,
            deadline
        };
        EvaluationBudget near_move_budget{
            0,
            kMaxNearCriticalMachineEvaluations,
            deadline
        };
        const Schedule near_sequence_schedule = FindBestSequenceNeighbor(
            schedule,
            jobs,
            job_list,
            near_sequence_processes,
            machines,
            near_sequence_budget
        );
        const Schedule near_move_schedule = FindBestMachineMoveNeighbor(
            schedule,
            jobs,
            job_list,
            near_move_processes,
            machines,
            near_move_budget
        );
        best_schedule = near_sequence_schedule;
        if (IsBetterSchedule(near_move_schedule, best_schedule)) {
            best_schedule = near_move_schedule;
        }
    }

    if (best_schedule.get_TotalTime() >= schedule.get_TotalTime() &&
        !SearchDeadlineReached(deadline)) {
        EvaluationBudget compound_budget{
            0,
            kMaxCompoundEvaluations,
            deadline
        };
        const Schedule compound_schedule = FindBestCompoundNeighbor(
            schedule,
            jobs,
            job_list,
            sequence_processes,
            machines,
            compound_budget
        );
        if (IsBetterSchedule(compound_schedule, best_schedule)) {
            best_schedule = compound_schedule;
        }
    }
    return best_schedule;
}

}  // namespace

void ResetGreedySearchState() {
    gNextCandidateWindow = 0;
    gActiveCandidateWindow = 0;
    gRepairFailures = 0;
}

Schedule GreedySearch(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const int iter_count,
    std::vector<Machine> &machines,
    const SearchDeadline deadline
) {
    Schedule current_schedule = schedule;
    if (iter_count <= 0 || SearchDeadlineReached(deadline)) {
        return current_schedule;
    }
    gActiveCandidateWindow = gNextCandidateWindow++;
    const bool use_rl_policy = GreedyRlPolicyEnabled();

    int neutral_move_count = 0;
    for (int iteration = 0; iteration < iter_count; ++iteration) {
        if (SearchDeadlineReached(deadline)) {
            break;
        }
        const std::vector<std::string> sequence_processes =
            ScoreProcessForExchange(current_schedule, jobs, jobList);
        const std::vector<std::string> move_processes =
            ScoreProcessForMove(current_schedule, jobs, jobList);
        if (sequence_processes.empty() && move_processes.empty()) {
            break;
        }

        Schedule best_neighbor = current_schedule;
        if (use_rl_policy) {
            best_neighbor = FindBestRlNeighbor(
                current_schedule,
                jobs,
                jobList,
                sequence_processes,
                move_processes,
                machines,
                true,
                true,
                false,
                kMaxSequenceEvaluationsPerStep +
                    kMaxMachineMoveEvaluationsPerStep,
                deadline
            );
        } else {
            EvaluationBudget sequence_budget{
                0,
                kMaxSequenceEvaluationsPerStep,
                deadline
            };
            EvaluationBudget move_budget{
                0,
                kMaxMachineMoveEvaluationsPerStep,
                deadline
            };
            const Schedule sequence_schedule = FindBestSequenceNeighbor(
                current_schedule,
                jobs,
                jobList,
                sequence_processes,
                machines,
                sequence_budget
            );
            const Schedule move_schedule = FindBestMachineMoveNeighbor(
                current_schedule,
                jobs,
                jobList,
                move_processes,
                machines,
                move_budget
            );
            best_neighbor = sequence_schedule;
            if (IsBetterSchedule(move_schedule, best_neighbor)) {
                best_neighbor = move_schedule;
            }
        }

        // Near-critical moves are substantially more expensive and rarely
        // help while an exact critical-path improvement still exists. Explore
        // them only after the focused neighborhood has reached a local optimum.
        if (best_neighbor.get_TotalTime() >= current_schedule.get_TotalTime()) {
            const auto near_sequence_processes = RankCriticalProcesses(
                current_schedule,
                jobs,
                false,
                true
            );
            const auto near_move_processes = RankCriticalProcesses(
                current_schedule,
                jobs,
                true,
                true
            );
            if (use_rl_policy) {
                best_neighbor = FindBestRlNeighbor(
                    current_schedule,
                    jobs,
                    jobList,
                    near_sequence_processes,
                    near_move_processes,
                    machines,
                    true,
                    true,
                    false,
                    kMaxNearCriticalSequenceEvaluations +
                        kMaxNearCriticalMachineEvaluations,
                    deadline
                );
            } else {
                EvaluationBudget near_sequence_budget{
                    0,
                    kMaxNearCriticalSequenceEvaluations,
                    deadline
                };
                EvaluationBudget near_move_budget{
                    0,
                    kMaxNearCriticalMachineEvaluations,
                    deadline
                };
                const Schedule near_sequence_schedule = FindBestSequenceNeighbor(
                    current_schedule,
                    jobs,
                    jobList,
                    near_sequence_processes,
                    machines,
                    near_sequence_budget
                );
                const Schedule near_move_schedule = FindBestMachineMoveNeighbor(
                    current_schedule,
                    jobs,
                    jobList,
                    near_move_processes,
                    machines,
                    near_move_budget
                );
                best_neighbor = near_sequence_schedule;
                if (IsBetterSchedule(near_move_schedule, best_neighbor)) {
                    best_neighbor = near_move_schedule;
                }
            }
        }

        if (best_neighbor.get_TotalTime() >= current_schedule.get_TotalTime()) {
            const auto compound_processes = ScoreProcessForExchange(
                current_schedule,
                jobs,
                jobList
            );
            Schedule compound_schedule = current_schedule;
            if (use_rl_policy) {
                compound_schedule = FindBestRlNeighbor(
                    current_schedule,
                    jobs,
                    jobList,
                    compound_processes,
                    {},
                    machines,
                    false,
                    false,
                    true,
                    kMaxCompoundEvaluations,
                    deadline
                );
            } else {
                EvaluationBudget compound_budget{
                    0,
                    kMaxCompoundEvaluations,
                    deadline
                };
                compound_schedule = FindBestCompoundNeighbor(
                    current_schedule,
                    jobs,
                    jobList,
                    compound_processes,
                    machines,
                    compound_budget
                );
            }
            if (IsBetterSchedule(compound_schedule, best_neighbor)) {
                best_neighbor = compound_schedule;
            }
        }

        const bool verify_with_baseline = use_rl_policy &&
            GetGreedyRlRuntimeConfig().verify_with_baseline;
        const bool fallback_after_miss = use_rl_policy &&
            GetGreedyRlRuntimeConfig().fallback_on_miss &&
            !IsBetterSchedule(best_neighbor, current_schedule);
        if ((verify_with_baseline || fallback_after_miss) &&
            !SearchDeadlineReached(deadline)) {
            if (verify_with_baseline) {
                RecordGreedyRlVerification();
            } else {
                RecordGreedyRlFallback();
            }
            const Schedule fallback_schedule = FindBestBaselineGreedyStep(
                current_schedule,
                jobs,
                jobList,
                machines,
                deadline
            );
            if (IsBetterSchedule(fallback_schedule, best_neighbor)) {
                best_neighbor = fallback_schedule;
            }
        }

        const char *flexible_repair_flag = std::getenv("HLS_GREEDY_FLEXIBLE_REPAIR");
        if (flexible_repair_flag != nullptr && std::string(flexible_repair_flag) == "1" &&
            best_neighbor.get_TotalTime() >= current_schedule.get_TotalTime() &&
            gActiveCandidateWindow % 2 == 0 && !SearchDeadlineReached(deadline)) {
            auto repair_processes = ScoreProcessForExchange(current_schedule, jobs, jobList);
            const auto near_processes = RankCriticalProcesses(current_schedule, jobs, false, true);
            repair_processes.insert(repair_processes.end(), near_processes.begin(), near_processes.end());
            if (!repair_processes.empty()) {
                const int offset = (gActiveCandidateWindow / 2 * 8) % repair_processes.size();
                std::rotate(repair_processes.begin(), repair_processes.begin() + offset, repair_processes.end());
                const char *relay_flag = std::getenv("HLS_GREEDY_REPAIR_RELAYS");
                const bool use_relays = relay_flag != nullptr && std::string(relay_flag) == "1" &&
                    gRepairFailures >= 2 && gActiveCandidateWindow % 4 == 0;
                const char *route_flag = std::getenv("HLS_GREEDY_ROUTE_REPAIR");
                const bool use_route_neighbors = !use_relays &&
                    route_flag != nullptr && std::string(route_flag) == "1" &&
                    IsRouteCoupledRepairProblem(jobs);
                if (use_route_neighbors) {
                    repair_processes = ExpandRepairProcessesWithJobNeighbors(
                        repair_processes,
                        jobs
                    );
                }
                if (use_relays) {
                    // Critical-only releases cannot exchange load with a busy
                    // alternative machine whose operations have positive slack.
                    // Include flexible partners on those machines in the SAME
                    // joint neighborhood, without requiring an improving first move.
                    const auto &items = current_schedule.get_schedule_items();
                    const auto locations = BuildProcessLocationMap(items);
                    const auto job_map = BuildJobIdMap(jobs);
                    const auto workloads = BuildMachineWorkloads(items, job_map);
                    std::unordered_set<int> target_machines, source_machines;
                    std::unordered_set<std::string> seen;
                    std::vector<std::string> combined;
                    const int anchor_count = std::min(8, static_cast<int>(repair_processes.size()));
                    for (int anchor = 0; anchor < anchor_count; ++anchor) {
                        const auto parsed = ParseProcessKey(repair_processes[anchor]);
                        if (!parsed || !job_map.contains(parsed->first)) continue;
                        const auto location = locations.find(BuildProcessId(parsed->first, parsed->second));
                        if (location == locations.end()) continue;
                        source_machines.insert(items[location->second.machine_index].machine_id);
                        const auto *process = FindJobProcess(*job_map.at(parsed->first), parsed->second);
                        if (process != nullptr) {
                            for (const auto &option : process->process_item) target_machines.insert(option.machine_id);
                        }
                        if (seen.insert(repair_processes[anchor]).second) combined.push_back(repair_processes[anchor]);
                    }
                    struct Relay { std::string key; int machine; int workload; int duration; bool reverse; };
                    std::vector<Relay> relays;
                    for (int machine = 0; machine < static_cast<int>(items.size()); ++machine) {
                        if (!target_machines.contains(items[machine].machine_id)) continue;
                        for (const auto &operation : items[machine].schedule_process) {
                            if (!job_map.contains(operation.job_id)) continue;
                            const auto *process = FindJobProcess(*job_map.at(operation.job_id), operation.process_id);
                            if (process == nullptr || process->process_item.size() < 2) continue;
                            bool reverse = false;
                            for (const auto &option : process->process_item) {
                                reverse = reverse || (option.machine_id != items[machine].machine_id &&
                                    source_machines.contains(option.machine_id));
                            }
                            relays.push_back({BuildProcessKey(operation.job_id, operation.process_id), machine,
                                workloads[machine], GetProcessingTime(*process, items[machine].machine_id), reverse});
                        }
                    }
                    std::sort(relays.begin(), relays.end(), [](const Relay &left, const Relay &right) {
                        return std::make_tuple(!left.reverse, -left.workload, -left.duration, left.key) <
                               std::make_tuple(!right.reverse, -right.workload, -right.duration, right.key);
                    });
                    std::unordered_map<int, int> per_machine;
                    for (const auto &relay : relays) {
                        if (per_machine[relay.machine] >= 3 || !seen.insert(relay.key).second) continue;
                        combined.push_back(relay.key);
                        ++per_machine[relay.machine];
                    }
                    for (const auto &key : repair_processes) if (seen.insert(key).second) combined.push_back(key);
                    repair_processes = std::move(combined);
                }
                Schedule repaired = RepairCoptFlexibleNeighborhood(
                    current_schedule, repair_processes, jobList, jobs, machines, deadline,
                    use_relays ? 0 : (use_route_neighbors ? 32 : 24),
                    use_relays ? 0.0 : (use_route_neighbors ? 5.0 : 3.0));
                gRepairFailures = repaired.get_TotalTime() < current_schedule.get_TotalTime() ? 0 : gRepairFailures + 1;
                if (repaired.get_TotalTime() < best_neighbor.get_TotalTime()) best_neighbor = std::move(repaired);
            }
        }

        if (!IsBetterSchedule(best_neighbor, current_schedule)) {
            const char *repair_flag = std::getenv("HLS_GREEDY_COPT_REPAIR");
            if (repair_flag != nullptr && std::string(repair_flag) == "1" &&
                gActiveCandidateWindow % 4 == 0 && !SearchDeadlineReached(deadline)) {
                auto repair_processes = ScoreProcessForExchange(current_schedule, jobs, jobList);
                const auto near_processes = RankCriticalProcesses(current_schedule, jobs, false, true);
                repair_processes.insert(repair_processes.end(), near_processes.begin(), near_processes.end());
                if (!repair_processes.empty()) {
                    const int offset = (gActiveCandidateWindow / 4 * 8) % repair_processes.size();
                    std::rotate(repair_processes.begin(), repair_processes.begin() + offset,
                                repair_processes.end());
                    best_neighbor = RepairCoptMachineSequence(current_schedule, repair_processes,
                                                             jobList, jobs, machines, deadline);
                    if (best_neighbor.get_TotalTime() < current_schedule.get_TotalTime()) {
                        current_schedule = std::move(best_neighbor);
                        neutral_move_count = 0;
                        continue;
                    }
                }
            }
            break;
        }
        if (best_neighbor.get_TotalTime() == current_schedule.get_TotalTime()) {
            if (neutral_move_count >= kMaximumNeutralMoves) {
                break;
            }
            ++neutral_move_count;
        } else {
            neutral_move_count = 0;
        }
        current_schedule = std::move(best_neighbor);
    }
    return current_schedule;
}

Schedule GreedyAdjust(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    std::vector<Machine> &machines
) {
    return GreedySearch(schedule, jobs, jobList, 1, machines);
}

bool hasCycle(const std::vector<std::vector<int>> &adjMatrix) {
    const int node_count = static_cast<int>(adjMatrix.size());
    std::vector<int> in_degree(node_count, 0);
    for (int from = 0; from < node_count; ++from) {
        for (int to = 0; to < node_count; ++to) {
            if (adjMatrix[from][to] != -1) {
                ++in_degree[to];
            }
        }
    }

    std::queue<int> ready_nodes;
    for (int node = 0; node < node_count; ++node) {
        if (in_degree[node] == 0) {
            ready_nodes.push(node);
        }
    }

    int visited_count = 0;
    while (!ready_nodes.empty()) {
        const int node = ready_nodes.front();
        ready_nodes.pop();
        ++visited_count;
        for (int next = 0; next < node_count; ++next) {
            if (adjMatrix[node][next] != -1 && --in_degree[next] == 0) {
                ready_nodes.push(next);
            }
        }
    }
    return visited_count != node_count;
}

Schedule ExchangeNeighborSearch(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    std::vector<std::string> &processlist,
    bool &flag,
    std::vector<Machine> &machines
) {
    EvaluationBudget budget{0, kMaxSequenceEvaluationsPerStep};
    Schedule result = FindBestSequenceNeighbor(
        schedule,
        jobs,
        jobList,
        processlist,
        machines,
        budget
    );
    flag = result.get_TotalTime() < schedule.get_TotalTime();
    if (!flag) {
        result.set_schedule_id(-1);
    }
    return result;
}

std::vector<std::string> ScoreProcessForExchange(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList
) {
    (void)jobList;
    return RankCriticalProcesses(schedule, jobs, false);
}

int GetItemIndex(const Schedule &schedule, const int job_id, const int process_id) {
    const auto location_map = BuildProcessLocationMap(schedule.get_schedule_items());
    const auto location_it = location_map.find(BuildProcessId(job_id, process_id));
    return location_it == location_map.end() ? -1 : location_it->second.item_index;
}

void GetMachineIdAndItemIdByProcess(
    const std::vector<Schedule_item> &schedule_items,
    const std::string &process,
    const int type,
    int &x,
    int &y
) {
    (void)type;
    x = -1;
    y = -1;
    const auto parsed_process = ParseProcessKey(process);
    if (!parsed_process.has_value()) {
        return;
    }
    const auto location_map = BuildProcessLocationMap(schedule_items);
    const auto location_it = location_map.find(
        BuildProcessId(parsed_process->first, parsed_process->second)
    );
    if (location_it != location_map.end()) {
        x = location_it->second.machine_index;
        y = location_it->second.item_index;
    }
}

Schedule MoveNeighborSearch(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    std::vector<std::string> &processlist,
    bool &flag,
    std::vector<Machine> &machines
) {
    EvaluationBudget budget{0, kMaxMachineMoveEvaluationsPerStep};
    Schedule result = FindBestMachineMoveNeighbor(
        schedule,
        jobs,
        jobList,
        processlist,
        machines,
        budget
    );
    flag = result.get_TotalTime() < schedule.get_TotalTime();
    if (!flag) {
        result.set_schedule_id(-1);
    }
    return result;
}

int GetStartTimeByProcess(const Schedule &schedule, const int job_id, const int process_id) {
    const auto start_time_map = BuildStartTimeMap(schedule);
    const auto start_time_it = start_time_map.find(BuildProcessId(job_id, process_id));
    return start_time_it == start_time_map.end() ? 0 : start_time_it->second;
}

std::vector<std::string> ScoreProcessForMove(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList
) {
    (void)jobList;
    return RankCriticalProcesses(schedule, jobs, true);
}

int GetDifferFromIdealMachine(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const int job_id,
    const int process_id
) {
    const auto job_id_map = BuildJobIdMap(jobs);
    const auto job_it = job_id_map.find(job_id);
    if (job_it == job_id_map.end()) {
        return 0;
    }
    const Job_process *job_process = FindJobProcess(*job_it->second, process_id);
    if (job_process == nullptr || job_process->process_item.empty()) {
        return 0;
    }

    int current_process_count = 0;
    int minimum_process_count = INT_MAX;
    const auto location_map = BuildProcessLocationMap(schedule.get_schedule_items());
    const auto location_it = location_map.find(BuildProcessId(job_id, process_id));
    if (location_it != location_map.end()) {
        current_process_count = schedule.get_schedule_items()[location_it->second.machine_index].process_count;
    }
    for (const auto &option : job_process->process_item) {
        minimum_process_count = std::min(
            minimum_process_count,
            GetProcessCountByMachineId(schedule, option.machine_id)
        );
    }
    return minimum_process_count == INT_MAX
        ? 0
        : current_process_count - minimum_process_count;
}

int GetProcessCountByMachineId(const Schedule &schedule, const int machine_id) {
    for (const auto &schedule_item : schedule.get_schedule_items()) {
        if (schedule_item.machine_id == machine_id) {
            return schedule_item.process_count;
        }
    }
    return 0;
}

Job SelectJobByJobId(const std::vector<Job> &jobs, const int job_id) {
    for (const auto &job : jobs) {
        if (job.get_job_id() == job_id) {
            return job;
        }
    }
    return Job();
}
