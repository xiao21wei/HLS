#include "GuidedSearch.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataProc.h"

namespace {

constexpr int kMaximumTrajectories = 6;
constexpr int kMaximumExactCandidates = 48;
constexpr int kMaximumCriticalProcesses = 32;
constexpr int kMaximumNearCriticalProcesses = 16;
constexpr int kMaximumRecentSchedules = 128;
constexpr int kMaximumNeutralMoves = 4;
constexpr int kMaximumDeterioratingMoves = 4;
constexpr int kTrajectoryRestartStagnation = 10;

struct ProcessLocation {
    int machine_index = -1;
    int position = -1;
};

struct InsertAction {
    int job_id = -1;
    int process_id = -1;
    int source_machine_id = -1;
    int target_machine_id = -1;
    int target_position = -1;
    std::string source_schedule_signature;
};

struct RankedAction {
    InsertAction action;
    double score = 0.0;
    int frequency = 0;
};

struct EvaluatedAction {
    InsertAction action;
    Schedule schedule;
    std::string signature;
};

struct Trajectory {
    Schedule current;
    Schedule best;
    int stagnation = 0;
    int neutral_moves = 0;
    int deteriorating_moves = 0;
    std::deque<std::string> recent_order;
    std::unordered_set<std::string> recent_signatures;
};

struct PersistentGuidedState {
    std::string problem_signature;
    std::vector<Trajectory> trajectories;
    std::unordered_map<std::string, int> action_frequency;
    int calls = 0;
    int next_trajectory = 0;
};

PersistentGuidedState &GetPersistentState() {
    static thread_local PersistentGuidedState state;
    return state;
}

std::mt19937 &GetGuidedRng() {
    static thread_local std::mt19937 random_engine(123456789U);
    return random_engine;
}

long long BuildProcessId(const int job_id, const int process_id) {
    return (static_cast<long long>(job_id) << 32) ^
           static_cast<unsigned int>(process_id);
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

std::string BuildScheduleSignature(
    const std::vector<Schedule_item> &schedule_items
) {
    std::string signature;
    for (const auto &schedule_item : schedule_items) {
        signature += std::to_string(schedule_item.machine_id);
        signature.push_back(':');
        for (const auto &process : schedule_item.schedule_process) {
            signature += std::to_string(process.job_id);
            signature.push_back('-');
            signature += std::to_string(process.process_id);
            signature.push_back(',');
        }
        signature.push_back('|');
    }
    return signature;
}

std::string BuildProblemSignature(
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list
) {
    std::string signature;
    for (const auto &job_name : job_list) {
        signature += job_name;
        signature.push_back('|');
    }
    for (const auto &job : jobs) {
        signature += std::to_string(job.get_job_id());
        signature.push_back(':');
        for (const auto &job_process : job.get_job_process()) {
            for (const auto &option : job_process.process_item) {
                signature += std::to_string(option.machine_id);
                signature.push_back('-');
                signature += std::to_string(option.process_time);
                signature.push_back(',');
            }
            signature.push_back(';');
        }
        signature.push_back('|');
    }
    return signature;
}

std::string BuildActionKey(const InsertAction &action) {
    return std::to_string(action.job_id) + ':' +
           std::to_string(action.process_id) + ':' +
           std::to_string(action.source_machine_id) + ':' +
           std::to_string(action.target_machine_id) + ':' +
           std::to_string(action.target_position);
}

std::unordered_map<long long, ProcessLocation> BuildLocationMap(
    const std::vector<Schedule_item> &schedule_items
) {
    std::unordered_map<long long, ProcessLocation> locations;
    for (int machine_index = 0;
         machine_index < static_cast<int>(schedule_items.size());
         ++machine_index) {
        const auto &processes = schedule_items[machine_index].schedule_process;
        for (int position = 0;
             position < static_cast<int>(processes.size());
             ++position) {
            locations[BuildProcessId(
                processes[position].job_id,
                processes[position].process_id
            )] = {machine_index, position};
        }
    }
    return locations;
}

std::unordered_map<int, int> BuildMachineIndexMap(
    const std::vector<Schedule_item> &schedule_items
) {
    std::unordered_map<int, int> result;
    for (int index = 0; index < static_cast<int>(schedule_items.size()); ++index) {
        result[schedule_items[index].machine_id] = index;
    }
    return result;
}

std::unordered_map<int, const Job *> BuildJobIdMap(
    const std::vector<Job> &jobs
) {
    std::unordered_map<int, const Job *> result;
    for (const auto &job : jobs) {
        result[job.get_job_id()] = &job;
    }
    return result;
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

int GetProcessingTime(
    const Job_process &job_process,
    const int machine_id
) {
    for (const auto &option : job_process.process_item) {
        if (option.machine_id == machine_id) {
            return option.process_time;
        }
    }
    return 0;
}

std::vector<int> BuildMachineWorkloads(
    const std::vector<Schedule_item> &schedule_items,
    const std::unordered_map<int, const Job *> &job_id_map
) {
    std::vector<int> workloads(schedule_items.size(), 0);
    for (int machine_index = 0;
         machine_index < static_cast<int>(schedule_items.size());
         ++machine_index) {
        for (const auto &process : schedule_items[machine_index].schedule_process) {
            const auto job_it = job_id_map.find(process.job_id);
            if (job_it == job_id_map.end()) {
                continue;
            }
            const Job_process *job_process = FindJobProcess(
                *job_it->second,
                process.process_id
            );
            if (job_process != nullptr) {
                workloads[machine_index] += GetProcessingTime(
                    *job_process,
                    schedule_items[machine_index].machine_id
                );
            }
        }
    }
    return workloads;
}

std::unordered_map<long long, int> BuildStartTimeMap(const Schedule &schedule) {
    std::unordered_map<long long, int> result;
    const auto &process_list = schedule.get_processList();
    const auto &start_times = schedule.get_start_time();
    const int count = std::min(process_list.size(), start_times.size());
    for (int index = 0; index < count; ++index) {
        const auto process = ParseProcessKey(process_list[index]);
        if (process.has_value()) {
            result[BuildProcessId(process->first, process->second)] =
                start_times[index];
        }
    }
    return result;
}

void RememberSchedule(Trajectory &trajectory, const std::string &signature) {
    if (!trajectory.recent_signatures.insert(signature).second) {
        return;
    }
    trajectory.recent_order.push_back(signature);
    while (static_cast<int>(trajectory.recent_order.size()) >
           kMaximumRecentSchedules) {
        trajectory.recent_signatures.erase(trajectory.recent_order.front());
        trajectory.recent_order.pop_front();
    }
}

Trajectory BuildTrajectory(const Schedule &schedule) {
    Trajectory trajectory;
    trajectory.current = schedule;
    trajectory.best = schedule;
    RememberSchedule(
        trajectory,
        BuildScheduleSignature(schedule.get_schedule_items())
    );
    return trajectory;
}

void InjectTrajectory(
    PersistentGuidedState &state,
    const Schedule &schedule
) {
    if (schedule.get_schedule_items().empty()) {
        return;
    }
    const std::string signature = BuildScheduleSignature(
        schedule.get_schedule_items()
    );
    for (const auto &trajectory : state.trajectories) {
        if (BuildScheduleSignature(trajectory.current.get_schedule_items()) ==
            signature) {
            return;
        }
    }
    if (static_cast<int>(state.trajectories.size()) < kMaximumTrajectories) {
        state.trajectories.push_back(BuildTrajectory(schedule));
        return;
    }
    const auto worst = std::max_element(
        state.trajectories.begin(),
        state.trajectories.end(),
        [](const Trajectory &left, const Trajectory &right) {
            return left.best.get_TotalTime() < right.best.get_TotalTime();
        }
    );
    if (worst != state.trajectories.end() &&
        schedule.get_TotalTime() < worst->best.get_TotalTime()) {
        *worst = BuildTrajectory(schedule);
    }
}

std::vector<std::pair<int, int>> SelectProcesses(
    const Schedule &schedule,
    const bool include_near_critical,
    const int call_index
) {
    const auto slack = schedule.GetProcessSlack();
    std::unordered_set<long long> exact_ids;
    std::vector<std::pair<int, int>> exact;
    std::vector<std::pair<int, int>> near;
    for (const auto &process_key : schedule.GetKeyProcess()) {
        const auto process = ParseProcessKey(process_key);
        if (!process.has_value()) {
            continue;
        }
        if (exact_ids.insert(BuildProcessId(
                process->first,
                process->second
            )).second) {
            exact.push_back(*process);
        }
    }
    if (include_near_critical) {
        const int slack_limit = std::max(
            6,
            std::min(20, schedule.get_TotalTime() / 40)
        );
        for (const auto &process_key : schedule.get_processList()) {
            const auto process = ParseProcessKey(process_key);
            const auto slack_it = slack.find(process_key);
            if (!process.has_value() || slack_it == slack.end() ||
                slack_it->second <= 0 || slack_it->second > slack_limit) {
                continue;
            }
            near.push_back(*process);
        }
    }

    auto rotate_and_limit = [call_index](
        std::vector<std::pair<int, int>> values,
        const int maximum
    ) {
        if (static_cast<int>(values.size()) > maximum) {
            const int offset = (call_index * maximum) % values.size();
            std::rotate(
                values.begin(),
                values.begin() + offset,
                values.end()
            );
            values.resize(maximum);
        }
        return values;
    };
    exact = rotate_and_limit(std::move(exact), kMaximumCriticalProcesses);
    near = rotate_and_limit(std::move(near), kMaximumNearCriticalProcesses);
    exact.insert(exact.end(), near.begin(), near.end());
    return exact;
}

std::vector<RankedAction> GenerateRankedActions(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::unordered_map<std::string, int> &action_frequency,
    const bool include_near_critical,
    const int call_index
) {
    const auto &schedule_items = schedule.get_schedule_items();
    const auto location_map = BuildLocationMap(schedule_items);
    const auto machine_index_map = BuildMachineIndexMap(schedule_items);
    const auto job_id_map = BuildJobIdMap(jobs);
    const auto workloads = BuildMachineWorkloads(schedule_items, job_id_map);
    const auto start_times = BuildStartTimeMap(schedule);
    const auto exact_keys = schedule.GetKeyProcess();
    std::unordered_set<long long> exact_ids;
    for (const auto &key : exact_keys) {
        const auto process = ParseProcessKey(key);
        if (process.has_value()) {
            exact_ids.insert(BuildProcessId(process->first, process->second));
        }
    }

    std::vector<RankedAction> actions;
    std::unordered_set<std::string> action_keys;
    const std::string source_signature = BuildScheduleSignature(schedule_items);
    for (const auto &[job_id, process_id] : SelectProcesses(
             schedule,
             include_near_critical,
             call_index
         )) {
        const long long identifier = BuildProcessId(job_id, process_id);
        const auto location_it = location_map.find(identifier);
        const auto job_it = job_id_map.find(job_id);
        if (location_it == location_map.end() || job_it == job_id_map.end()) {
            continue;
        }
        const Job_process *job_process = FindJobProcess(
            *job_it->second,
            process_id
        );
        if (job_process == nullptr || job_process->process_item.size() <= 1) {
            continue;
        }
        const int source_machine_index = location_it->second.machine_index;
        const int source_machine_id = schedule_items[source_machine_index].machine_id;
        const int current_processing_time = GetProcessingTime(
            *job_process,
            source_machine_id
        );
        const int current_start = start_times.contains(identifier)
            ? start_times.at(identifier)
            : 0;

        for (const auto &option : job_process->process_item) {
            if (option.machine_id == source_machine_id) {
                continue;
            }
            const auto target_it = machine_index_map.find(option.machine_id);
            if (target_it == machine_index_map.end()) {
                continue;
            }
            const int target_machine_index = target_it->second;
            const auto &target_processes =
                schedule_items[target_machine_index].schedule_process;
            int preferred_position = static_cast<int>(target_processes.size());
            for (int position = 0;
                 position < static_cast<int>(target_processes.size());
                 ++position) {
                const auto &target_process = target_processes[position];
                const auto target_start = start_times.find(BuildProcessId(
                    target_process.job_id,
                    target_process.process_id
                ));
                if (target_start != start_times.end() &&
                    target_start->second >= current_start) {
                    preferred_position = position;
                    break;
                }
            }

            for (int target_position = 0;
                 target_position <= static_cast<int>(target_processes.size());
                 ++target_position) {
                InsertAction action{
                    job_id,
                    process_id,
                    source_machine_id,
                    option.machine_id,
                    target_position,
                    source_signature
                };
                const std::string action_key = BuildActionKey(action);
                if (!action_keys.insert(action_key).second) {
                    continue;
                }
                const auto frequency_it = action_frequency.find(action_key);
                const int frequency = frequency_it == action_frequency.end()
                    ? 0
                    : frequency_it->second;
                const double processing_gain =
                    current_processing_time - option.process_time;
                const double load_gain =
                    workloads[source_machine_index] -
                    workloads[target_machine_index];
                const double position_distance = std::abs(
                    target_position - preferred_position
                );
                const double exact_bonus = exact_ids.contains(identifier)
                    ? 20.0
                    : 0.0;
                const double score = exact_bonus +
                    8.0 * processing_gain +
                    0.20 * load_gain -
                    0.75 * position_distance -
                    2.0 * frequency;
                actions.push_back({std::move(action), score, frequency});
            }
        }
    }
    return actions;
}

std::vector<RankedAction> SelectActionsForExactEvaluation(
    std::vector<RankedAction> actions,
    const int maximum
) {
    std::stable_sort(
        actions.begin(),
        actions.end(),
        [](const RankedAction &left, const RankedAction &right) {
            if (left.score != right.score) {
                return left.score > right.score;
            }
            return left.frequency < right.frequency;
        }
    );
    if (static_cast<int>(actions.size()) <= maximum) {
        return actions;
    }

    std::vector<RankedAction> selected;
    selected.reserve(maximum);
    std::unordered_set<std::string> selected_keys;
    std::unordered_set<long long> selected_processes;
    std::unordered_set<int> selected_machines;
    auto add_action = [&](const RankedAction &action) {
        const std::string key = BuildActionKey(action.action);
        if (!selected_keys.insert(key).second) {
            return false;
        }
        selected.push_back(action);
        return true;
    };

    const int process_quota = maximum / 3;
    for (const auto &action : actions) {
        const long long process_id = BuildProcessId(
            action.action.job_id,
            action.action.process_id
        );
        if (selected_processes.insert(process_id).second) {
            add_action(action);
        }
        if (static_cast<int>(selected.size()) >= process_quota) {
            break;
        }
    }

    const int machine_quota = maximum / 2;
    for (const auto &action : actions) {
        if (selected_machines.insert(action.action.target_machine_id).second) {
            add_action(action);
        }
        if (static_cast<int>(selected.size()) >= machine_quota) {
            break;
        }
    }

    std::vector<int> remaining_indices;
    for (int index = 0; index < static_cast<int>(actions.size()); ++index) {
        if (!selected_keys.contains(BuildActionKey(actions[index].action))) {
            remaining_indices.push_back(index);
        }
    }
    std::shuffle(
        remaining_indices.begin(),
        remaining_indices.end(),
        GetGuidedRng()
    );
    const int exploration_target = std::min(
        maximum,
        static_cast<int>(selected.size()) + std::max(4, maximum / 8)
    );
    for (const int index : remaining_indices) {
        add_action(actions[index]);
        if (static_cast<int>(selected.size()) >= exploration_target) {
            break;
        }
    }
    for (const auto &action : actions) {
        add_action(action);
        if (static_cast<int>(selected.size()) >= maximum) {
            break;
        }
    }
    return selected;
}

std::optional<EvaluatedAction> EvaluateAction(
    const Schedule &base_schedule,
    const InsertAction &action,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    std::vector<Machine> &machines
) {
    const auto &base_items = base_schedule.get_schedule_items();
    if (BuildScheduleSignature(base_items) != action.source_schedule_signature) {
        return std::nullopt;
    }
    const auto location_map = BuildLocationMap(base_items);
    const auto machine_index_map = BuildMachineIndexMap(base_items);
    const auto source_it = location_map.find(BuildProcessId(
        action.job_id,
        action.process_id
    ));
    const auto target_it = machine_index_map.find(action.target_machine_id);
    if (source_it == location_map.end() ||
        target_it == machine_index_map.end()) {
        return std::nullopt;
    }
    const int source_machine_index = source_it->second.machine_index;
    const int target_machine_index = target_it->second;
    if (base_items[source_machine_index].machine_id != action.source_machine_id ||
        source_machine_index == target_machine_index) {
        return std::nullopt;
    }

    std::vector<Schedule_item> candidate_items = base_items;
    auto &source_processes =
        candidate_items[source_machine_index].schedule_process;
    if (source_it->second.position < 0 ||
        source_it->second.position >= static_cast<int>(source_processes.size())) {
        return std::nullopt;
    }
    const Schedule_process moved_process =
        source_processes[source_it->second.position];
    source_processes.erase(
        source_processes.begin() + source_it->second.position
    );
    candidate_items[source_machine_index].process_count =
        static_cast<int>(source_processes.size());

    auto &target_processes =
        candidate_items[target_machine_index].schedule_process;
    if (action.target_position < 0 ||
        action.target_position > static_cast<int>(target_processes.size())) {
        return std::nullopt;
    }
    target_processes.insert(
        target_processes.begin() + action.target_position,
        moved_process
    );
    candidate_items[target_machine_index].process_count =
        static_cast<int>(target_processes.size());

    Schedule candidate = base_schedule;
    candidate.set_schedule_id(base_schedule.get_schedule_id() + 1);
    candidate.set_schedule_items(candidate_items);
    try {
        ScheduleItemsToGraph(
            candidate,
            candidate_items,
            jobs,
            job_list,
            true,
            machines
        );
    } catch (const std::runtime_error &) {
        return std::nullopt;
    }
    return EvaluatedAction{
        action,
        std::move(candidate),
        BuildScheduleSignature(candidate_items)
    };
}

bool IsBetterEvaluatedAction(
    const EvaluatedAction &left,
    const EvaluatedAction &right,
    const std::unordered_map<std::string, int> &frequency
) {
    if (left.schedule.get_TotalTime() != right.schedule.get_TotalTime()) {
        return left.schedule.get_TotalTime() < right.schedule.get_TotalTime();
    }
    const int left_frequency = frequency.contains(BuildActionKey(left.action))
        ? frequency.at(BuildActionKey(left.action))
        : 0;
    const int right_frequency = frequency.contains(BuildActionKey(right.action))
        ? frequency.at(BuildActionKey(right.action))
        : 0;
    if (left_frequency != right_frequency) {
        return left_frequency < right_frequency;
    }
    return left.signature < right.signature;
}

bool AdvanceTrajectory(
    Trajectory &trajectory,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    std::vector<Machine> &machines,
    PersistentGuidedState &state,
    const SearchDeadline deadline
) {
    const bool include_near_critical = trajectory.stagnation >= 4;
    auto actions = GenerateRankedActions(
        trajectory.current,
        jobs,
        state.action_frequency,
        include_near_critical,
        state.calls + trajectory.stagnation
    );
    actions = SelectActionsForExactEvaluation(
        std::move(actions),
        kMaximumExactCandidates
    );

    std::optional<EvaluatedAction> best_candidate;
    for (const auto &ranked_action : actions) {
        if (SearchDeadlineReached(deadline)) {
            break;
        }
        auto candidate = EvaluateAction(
            trajectory.current,
            ranked_action.action,
            jobs,
            job_list,
            machines
        );
        if (!candidate.has_value()) {
            continue;
        }
        const bool improves_best =
            candidate->schedule.get_TotalTime() <
                trajectory.best.get_TotalTime();
        if (!improves_best &&
            trajectory.recent_signatures.contains(candidate->signature)) {
            continue;
        }
        if (!best_candidate.has_value() ||
            IsBetterEvaluatedAction(
                *candidate,
                *best_candidate,
                state.action_frequency
            )) {
            best_candidate = std::move(candidate);
        }
    }

    if (!best_candidate.has_value()) {
        ++trajectory.stagnation;
        return false;
    }
    const int delta = best_candidate->schedule.get_TotalTime() -
                      trajectory.current.get_TotalTime();
    const int maximum_deterioration = std::max(
        3,
        static_cast<int>(std::lround(
            trajectory.best.get_TotalTime() * 0.02
        ))
    );
    bool accept = delta < 0;
    if (delta == 0 &&
        trajectory.neutral_moves < kMaximumNeutralMoves) {
        accept = true;
    }
    if (delta > 0 && trajectory.stagnation >= 4 &&
        trajectory.deteriorating_moves < kMaximumDeterioratingMoves &&
        delta <= maximum_deterioration) {
        accept = true;
    }
    if (!accept) {
        ++trajectory.stagnation;
        return false;
    }

    ++state.action_frequency[BuildActionKey(best_candidate->action)];
    trajectory.current = std::move(best_candidate->schedule);
    RememberSchedule(trajectory, best_candidate->signature);
    if (delta == 0) {
        ++trajectory.neutral_moves;
    } else if (delta > 0) {
        ++trajectory.deteriorating_moves;
    } else {
        trajectory.neutral_moves = 0;
        trajectory.deteriorating_moves = 0;
    }
    if (trajectory.current.get_TotalTime() <
        trajectory.best.get_TotalTime()) {
        trajectory.best = trajectory.current;
        trajectory.stagnation = 0;
    } else {
        ++trajectory.stagnation;
    }
    return true;
}

}  // namespace

void SetGuidedSearchSeed(const unsigned int random_seed) {
    GetGuidedRng().seed(random_seed);
}

Schedule GuidedInsertionSearch(
    const Schedule &schedule,
    const std::vector<Schedule> &seed_schedules,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    std::vector<Machine> &machines,
    const int iteration_count,
    const SearchDeadline deadline
) {
    if (iteration_count <= 0 || schedule.get_schedule_items().empty() ||
        SearchDeadlineReached(deadline)) {
        return schedule;
    }

    PersistentGuidedState &state = GetPersistentState();
    const std::string problem_signature = BuildProblemSignature(jobs, job_list);
    if (state.problem_signature != problem_signature) {
        state = PersistentGuidedState{};
        state.problem_signature = problem_signature;
    }
    InjectTrajectory(state, schedule);
    for (const auto &seed_schedule : seed_schedules) {
        InjectTrajectory(state, seed_schedule);
    }
    if (state.trajectories.empty()) {
        return schedule;
    }

    Schedule best_schedule = schedule;
    for (const auto &trajectory : state.trajectories) {
        if (trajectory.best.get_TotalTime() < best_schedule.get_TotalTime()) {
            best_schedule = trajectory.best;
        }
    }

    for (int iteration = 0;
         iteration < iteration_count && !SearchDeadlineReached(deadline);
         ++iteration) {
        const int trajectory_index = state.next_trajectory %
            static_cast<int>(state.trajectories.size());
        state.next_trajectory = (trajectory_index + 1) %
            static_cast<int>(state.trajectories.size());
        Trajectory &trajectory = state.trajectories[trajectory_index];
        AdvanceTrajectory(
            trajectory,
            jobs,
            job_list,
            machines,
            state,
            deadline
        );
        if (trajectory.best.get_TotalTime() < best_schedule.get_TotalTime()) {
            best_schedule = trajectory.best;
        }
        if (trajectory.stagnation >= kTrajectoryRestartStagnation) {
            trajectory.current = trajectory.best;
            trajectory.stagnation = 0;
            trajectory.neutral_moves = 0;
            trajectory.deteriorating_moves = 0;
            trajectory.recent_order.clear();
            trajectory.recent_signatures.clear();
            RememberSchedule(
                trajectory,
                BuildScheduleSignature(
                    trajectory.current.get_schedule_items()
                )
            );
        }
    }

    ++state.calls;
    if (state.calls % 20 == 0) {
        for (auto it = state.action_frequency.begin();
             it != state.action_frequency.end();) {
            it->second /= 2;
            if (it->second == 0) {
                it = state.action_frequency.erase(it);
            } else {
                ++it;
            }
        }
    }
    CalculateTotalFailureRate(best_schedule, machines);
    return best_schedule;
}
