//
// Created by luopw on 24-12-1.
//

#include "GreedySearch.h"

#include <algorithm>
#include <climits>
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

bool EvaluateCandidate(
    const Schedule &base_schedule,
    std::vector<Schedule_item> schedule_items,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    std::vector<Machine> &machines,
    Schedule &candidate
) {
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
            candidate
        )) {
        return;
    }

    if (IsBetterSchedule(candidate, best_schedule)) {
        CalculateTotalFailureRate(candidate, machines);
        best_schedule = std::move(candidate);
    }
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
                    candidate
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

}  // namespace

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

        Schedule best_neighbor = sequence_schedule;
        if (IsBetterSchedule(move_schedule, best_neighbor)) {
            best_neighbor = move_schedule;
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

        if (best_neighbor.get_TotalTime() >= current_schedule.get_TotalTime()) {
            const auto compound_processes = ScoreProcessForExchange(
                current_schedule,
                jobs,
                jobList
            );
            EvaluationBudget compound_budget{
                0,
                kMaxCompoundEvaluations,
                deadline
            };
            const Schedule compound_schedule = FindBestCompoundNeighbor(
                current_schedule,
                jobs,
                jobList,
                compound_processes,
                machines,
                compound_budget
            );
            if (IsBetterSchedule(compound_schedule, best_neighbor)) {
                best_neighbor = compound_schedule;
            }
        }

        if (!IsBetterSchedule(best_neighbor, current_schedule)) {
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
