//
// Created by lpw on 25-4-14.
//

#include "TabuSearch.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataProc.h"
#include "GreedyRlPolicy.h"
#include "Job.h"
#include "Schedule.h"
#include "TabuRlPolicy.h"

namespace {

constexpr int kMaxCriticalProcesses = 36;
constexpr int kMaxNeighborEvaluations = 2600;
constexpr int kMaxCriticalBlockEvaluations = 400;
constexpr int kMaxMachineMoveEvaluations = 1100;
constexpr int kMaxSequenceMoveEvaluations = 650;
constexpr int kDiversificationPoolSize = 12;
constexpr long long kNoPredecessor = -1;
constexpr long long kNoSuccessor = -2;

std::mt19937 &GetTabuRng() {
    static thread_local std::mt19937 random_engine(362436069U);
    return random_engine;
}

int RandomInteger(const int lower, const int upper) {
    return std::uniform_int_distribution<int>(lower, upper)(GetTabuRng());
}

int EnvironmentInteger(
    const char *name,
    const int fallback,
    const int minimum,
    const int maximum
) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    try {
        return std::clamp(std::stoi(value), minimum, maximum);
    } catch (const std::exception &) {
        return fallback;
    }
}

bool SparseEvaluatorVerificationEnabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("HLS_VERIFY_SPARSE_EVALUATOR");
        return value != nullptr && std::string(value) != "0" &&
               std::string(value) != "false";
    }();
    return enabled;
}

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

struct ParsedProcess {
    int job_id;
    int process_id;
    long long identifier;
};

struct CriticalProcesses {
    std::vector<ParsedProcess> ordered;
    std::unordered_set<long long> exact_identifiers;
};

struct CriticalBlock {
    int machine_index;
    int first_position;
    int last_position;
};

struct MoveAttribute {
    long long process_identifier = 0;
    int source_machine_id = -1;
    int target_machine_id = -1;
    long long old_predecessor = kNoPredecessor;
    long long old_successor = kNoSuccessor;
    long long new_predecessor = kNoPredecessor;
    long long new_successor = kNoSuccessor;
    std::vector<std::string> extra_destination_attributes;
    std::vector<std::string> extra_reverse_attributes;
};

enum class TabuNeighborhoodType {
    CriticalBlock = 0,
    PathRelinking = 1,
    MachineMove = 2,
    Compound = 3,
    SequenceMove = 4
};

constexpr int kTabuNeighborhoodTypeCount = 5;

struct TabuRlUnevaluatedCandidate {
    std::vector<Schedule_item> schedule_items;
    MoveAttribute move;
    std::string signature;
    GreedyRlFeatures features{};
    TabuNeighborhoodType neighborhood = TabuNeighborhoodType::SequenceMove;
};

struct CandidateRecord {
    std::vector<Schedule_item> schedule_items;
    MoveAttribute move;
    std::string signature;
    int makespan = 0;
    double selection_score = 0.0;
};

struct PersistentTabuState {
    std::string problem_signature;
    std::unordered_map<std::string, int> move_frequency;
    std::unordered_map<std::string, int> tabu_expiration;
    std::optional<Schedule> current_schedule;
    int absolute_iteration = 0;
};

PersistentTabuState &GetPersistentTabuState() {
    static thread_local PersistentTabuState state;
    return state;
}

struct SelectionResult {
    std::optional<CandidateRecord> candidate;
    bool forced_tabu_move = false;
};

using CandidateVisitor = std::function<bool(
    std::vector<Schedule_item> &&,
    const MoveAttribute &
)>;

std::unordered_map<long long, ProcessLocation> BuildProcessLocationMap(
    const std::vector<Schedule_item> &schedule_items
) {
    std::unordered_map<long long, ProcessLocation> location_map;
    int process_count = 0;
    for (const auto &schedule_item : schedule_items) {
        process_count += static_cast<int>(schedule_item.schedule_process.size());
    }
    location_map.reserve(process_count);

    for (int machine_index = 0;
         machine_index < static_cast<int>(schedule_items.size());
         ++machine_index) {
        const auto &processes = schedule_items[machine_index].schedule_process;
        for (int item_index = 0;
             item_index < static_cast<int>(processes.size());
             ++item_index) {
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

std::vector<int> BuildMachineWorkloads(
    const std::vector<Schedule_item> &schedule_items,
    const std::unordered_map<int, const Job *> &job_id_map
) {
    std::vector<int> workloads(schedule_items.size(), 0);
    for (int machine_index = 0;
         machine_index < static_cast<int>(schedule_items.size());
         ++machine_index) {
        const auto &schedule_item = schedule_items[machine_index];
        for (const auto &process : schedule_item.schedule_process) {
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
                    schedule_item.machine_id
                );
            }
        }
    }
    return workloads;
}

int CountOperations(const std::vector<Schedule_item> &schedule_items) {
    int operation_count = 0;
    for (const auto &schedule_item : schedule_items) {
        operation_count += static_cast<int>(schedule_item.schedule_process.size());
    }
    return operation_count;
}

std::string BuildProblemSignature(
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list
) {
    std::string signature;
    signature.reserve(jobs.size() * 32 + job_list.size() * 12);
    for (const auto &job_name : job_list) {
        signature += job_name;
        signature.push_back('|');
    }
    for (const auto &job : jobs) {
        signature += std::to_string(job.get_job_id());
        signature.push_back(':');
        signature += std::to_string(job.get_process_count());
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

std::vector<std::string> BuildProcessListFromScheduleItems(
    const std::vector<Schedule_item> &schedule_items
) {
    std::vector<std::string> process_list;
    process_list.reserve(CountOperations(schedule_items) + 2);
    process_list.push_back("start");
    for (const auto &schedule_item : schedule_items) {
        for (const auto &process : schedule_item.schedule_process) {
            process_list.push_back(BuildProcessKey(process.job_id, process.process_id));
        }
    }
    process_list.push_back("end");
    return process_list;
}

std::string BuildScheduleSignature(const std::vector<Schedule_item> &schedule_items) {
    std::string signature;
    signature.reserve(static_cast<size_t>(CountOperations(schedule_items)) * 18 +
                      schedule_items.size() * 8);
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

CriticalProcesses BuildCriticalProcesses(
    const Schedule &schedule,
    const bool include_near_critical
) {
    CriticalProcesses critical_processes;
    const auto process_slack = schedule.GetProcessSlack();
    const int near_critical_limit = std::max(
        6,
        std::min(20, schedule.get_TotalTime() / 40)
    );
    const std::vector<std::string> key_processes = schedule.GetKeyProcess();
    critical_processes.ordered.reserve(schedule.get_processList().size());
    critical_processes.exact_identifiers.reserve(key_processes.size());
    std::unordered_set<long long> seen_identifiers;
    seen_identifiers.reserve(schedule.get_processList().size());

    auto add_process = [&](const std::string &process_key, const bool exact) {
        const auto parsed_process = ParseProcessKey(process_key);
        if (!parsed_process.has_value()) {
            return;
        }
        const auto slack_it = process_slack.find(process_key);
        if (slack_it == process_slack.end() ||
            slack_it->second > near_critical_limit) {
            return;
        }
        const long long identifier = BuildProcessId(
            parsed_process->first,
            parsed_process->second
        );
        if (!seen_identifiers.insert(identifier).second) {
            return;
        }
        if (exact) {
            critical_processes.exact_identifiers.insert(identifier);
        }
        critical_processes.ordered.push_back({
            parsed_process->first,
            parsed_process->second,
            identifier
        });
    };

    // Keep exact critical operations first, then add near-critical operations
    // so the neighborhood can escape a flat local optimum.
    for (const auto &process_key : key_processes) {
        add_process(process_key, true);
    }
    if (include_near_critical) {
        for (const auto &process_key : schedule.get_processList()) {
            add_process(process_key, false);
        }
    }
    return critical_processes;
}

void RotateCriticalCandidates(
    CriticalProcesses &critical_processes,
    const int iteration
) {
    if (static_cast<int>(critical_processes.ordered.size()) <=
        kMaxCriticalProcesses) {
        return;
    }
    const int offset = static_cast<int>(
        (static_cast<long long>(iteration) * kMaxCriticalProcesses) %
        critical_processes.ordered.size()
    );
    std::rotate(
        critical_processes.ordered.begin(),
        critical_processes.ordered.begin() + offset,
        critical_processes.ordered.end()
    );
}

std::unordered_map<long long, int> BuildStartTimeMap(const Schedule &schedule) {
    std::unordered_map<long long, int> start_time_map;
    const auto &process_list = schedule.get_processList();
    const auto &start_times = schedule.get_start_time();
    const int count = std::min(
        static_cast<int>(process_list.size()),
        static_cast<int>(start_times.size())
    );
    start_time_map.reserve(count);

    for (int index = 0; index < count; ++index) {
        const auto parsed_process = ParseProcessKey(process_list[index]);
        if (parsed_process.has_value()) {
            start_time_map[BuildProcessId(
                parsed_process->first,
                parsed_process->second
            )] = start_times[index];
        }
    }
    return start_time_map;
}

std::vector<CriticalBlock> BuildCriticalBlocks(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const CriticalProcesses &critical_processes
) {
    std::vector<CriticalBlock> blocks;
    const auto &schedule_items = schedule.get_schedule_items();
    const auto start_time_map = BuildStartTimeMap(schedule);
    const auto job_id_map = BuildJobIdMap(jobs);

    for (int machine_index = 0;
         machine_index < static_cast<int>(schedule_items.size());
         ++machine_index) {
        const auto &schedule_item = schedule_items[machine_index];
        const auto &processes = schedule_item.schedule_process;
        int block_start = -1;

        for (int position = 0;
             position + 1 < static_cast<int>(processes.size());
             ++position) {
            const auto &left = processes[position];
            const auto &right = processes[position + 1];
            const long long left_id = BuildProcessId(left.job_id, left.process_id);
            const long long right_id = BuildProcessId(right.job_id, right.process_id);

            bool is_tight_critical_arc =
                critical_processes.exact_identifiers.contains(left_id) &&
                critical_processes.exact_identifiers.contains(right_id);
            const auto left_start_it = start_time_map.find(left_id);
            const auto right_start_it = start_time_map.find(right_id);
            const auto job_it = job_id_map.find(left.job_id);
            if (is_tight_critical_arc &&
                left_start_it != start_time_map.end() &&
                right_start_it != start_time_map.end() &&
                job_it != job_id_map.end()) {
                const Job_process *job_process = FindJobProcess(
                    *job_it->second,
                    left.process_id
                );
                is_tight_critical_arc =
                    job_process != nullptr &&
                    right_start_it->second ==
                        left_start_it->second +
                        GetProcessingTime(*job_process, schedule_item.machine_id);
            } else {
                is_tight_critical_arc = false;
            }

            if (is_tight_critical_arc) {
                if (block_start < 0) {
                    block_start = position;
                }
            } else if (block_start >= 0) {
                blocks.push_back({machine_index, block_start, position});
                block_start = -1;
            }
        }

        if (block_start >= 0) {
            blocks.push_back({
                machine_index,
                block_start,
                static_cast<int>(processes.size()) - 1
            });
        }
    }
    return blocks;
}

long long GetNeighborIdentifier(
    const std::vector<Schedule_process> &processes,
    const int position,
    const bool predecessor
) {
    const int neighbor_position = predecessor ? position - 1 : position + 1;
    if (neighbor_position < 0) {
        return kNoPredecessor;
    }
    if (neighbor_position >= static_cast<int>(processes.size())) {
        return kNoSuccessor;
    }
    const auto &neighbor = processes[neighbor_position];
    return BuildProcessId(neighbor.job_id, neighbor.process_id);
}

std::optional<MoveAttribute> DescribeMove(
    const std::vector<Schedule_item> &original_items,
    const std::vector<Schedule_item> &candidate_items,
    const Schedule_process &moved_process
) {
    const long long process_identifier = BuildProcessId(
        moved_process.job_id,
        moved_process.process_id
    );
    const auto original_location_map = BuildProcessLocationMap(original_items);
    const auto candidate_location_map = BuildProcessLocationMap(candidate_items);
    const auto original_it = original_location_map.find(process_identifier);
    const auto candidate_it = candidate_location_map.find(process_identifier);
    if (original_it == original_location_map.end() ||
        candidate_it == candidate_location_map.end()) {
        return std::nullopt;
    }

    const auto &original_processes =
        original_items[original_it->second.machine_index].schedule_process;
    const auto &candidate_processes =
        candidate_items[candidate_it->second.machine_index].schedule_process;
    return MoveAttribute{
        process_identifier,
        original_items[original_it->second.machine_index].machine_id,
        candidate_items[candidate_it->second.machine_index].machine_id,
        GetNeighborIdentifier(original_processes, original_it->second.item_index, true),
        GetNeighborIdentifier(original_processes, original_it->second.item_index, false),
        GetNeighborIdentifier(candidate_processes, candidate_it->second.item_index, true),
        GetNeighborIdentifier(candidate_processes, candidate_it->second.item_index, false),
        {},
        {}
    };
}

std::string BuildPositionAttributeKey(
    const long long process_identifier,
    const int machine_id,
    const long long predecessor,
    const long long successor
) {
    return "P:" + std::to_string(process_identifier) + ":" +
           std::to_string(machine_id) + ":" +
           std::to_string(predecessor) + ":" +
           std::to_string(successor);
}

std::string BuildMachineAttributeKey(
    const long long process_identifier,
    const int machine_id
) {
    return "M:" + std::to_string(process_identifier) + ":" +
           std::to_string(machine_id);
}

std::vector<std::string> BuildDestinationAttributes(const MoveAttribute &move) {
    std::vector<std::string> attributes;
    attributes.reserve(2 + move.extra_destination_attributes.size());
    attributes.push_back(BuildPositionAttributeKey(
        move.process_identifier,
        move.target_machine_id,
        move.new_predecessor,
        move.new_successor
    ));
    if (move.source_machine_id != move.target_machine_id) {
        attributes.push_back(BuildMachineAttributeKey(
            move.process_identifier,
            move.target_machine_id
        ));
    }
    attributes.insert(
        attributes.end(),
        move.extra_destination_attributes.begin(),
        move.extra_destination_attributes.end()
    );
    return attributes;
}

std::vector<std::string> BuildReverseAttributes(const MoveAttribute &move) {
    std::vector<std::string> attributes;
    attributes.reserve(2 + move.extra_reverse_attributes.size());
    attributes.push_back(BuildPositionAttributeKey(
        move.process_identifier,
        move.source_machine_id,
        move.old_predecessor,
        move.old_successor
    ));
    if (move.source_machine_id != move.target_machine_id) {
        attributes.push_back(BuildMachineAttributeKey(
            move.process_identifier,
            move.source_machine_id
        ));
    }
    attributes.insert(
        attributes.end(),
        move.extra_reverse_attributes.begin(),
        move.extra_reverse_attributes.end()
    );
    return attributes;
}

std::string BuildFrequencyKey(const MoveAttribute &move) {
    return BuildPositionAttributeKey(
        move.process_identifier,
        move.target_machine_id,
        move.new_predecessor,
        move.new_successor
    );
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

bool IsMoveTabu(
    const MoveAttribute &move,
    const std::unordered_map<std::string, int> &tabu_expiration,
    const int iteration
) {
    for (const auto &attribute : BuildDestinationAttributes(move)) {
        const auto tabu_it = tabu_expiration.find(attribute);
        if (tabu_it != tabu_expiration.end() && tabu_it->second > iteration) {
            return true;
        }
    }
    return false;
}

bool IsBetterCandidate(
    const CandidateRecord &left,
    const CandidateRecord &right
) {
    if (left.selection_score != right.selection_score) {
        return left.selection_score < right.selection_score;
    }
    if (left.makespan != right.makespan) {
        return left.makespan < right.makespan;
    }
    return left.signature < right.signature;
}

class NeighborEvaluator {
public:
    NeighborEvaluator(
        const Schedule &base_schedule,
        const std::vector<Job> &jobs,
        const std::vector<std::string> &job_list,
        std::vector<Machine> &machines,
        const std::unordered_map<std::string, int> &tabu_expiration,
        const std::unordered_map<std::string, int> &move_frequency,
        const int iteration,
        const int global_best_makespan,
        const int stagnation_count,
        const bool diversification_mode,
        const SearchDeadline deadline,
        const int max_deterioration
    ) : base_schedule_(base_schedule),
        jobs_(jobs),
        job_list_(job_list),
        sparse_evaluator_(jobs, job_list),
        machines_(machines),
        tabu_expiration_(tabu_expiration),
        move_frequency_(move_frequency),
        iteration_(iteration),
        global_best_makespan_(global_best_makespan),
        stagnation_count_(stagnation_count),
        diversification_mode_(diversification_mode),
        deadline_(deadline),
        max_deterioration_(max_deterioration) {
        seen_signatures_.reserve(kMaxNeighborEvaluations * 2);
        seen_signatures_.insert(BuildScheduleSignature(base_schedule.get_schedule_items()));
    }

    bool Consider(
        std::vector<Schedule_item> &&schedule_items,
        const MoveAttribute &move,
        std::string signature = {}
    ) {
        if (!CanContinue()) {
            return false;
        }

        if (signature.empty()) {
            signature = BuildScheduleSignature(schedule_items);
        }
        if (!seen_signatures_.insert(signature).second) {
            return true;
        }
        ++evaluated_count_;

        int candidate_makespan = 0;
        if (!sparse_evaluator_.Evaluate(
                schedule_items,
                candidate_makespan
            )) {
            return CanContinue();
        }
        if (SparseEvaluatorVerificationEnabled()) {
            Schedule dense_candidate;
            if (!EvaluateCandidate(
                    base_schedule_,
                    schedule_items,
                    jobs_,
                    job_list_,
                    machines_,
                    dense_candidate
                ) ||
                dense_candidate.get_TotalTime() != candidate_makespan) {
                throw std::runtime_error(
                    "sparse Tabu evaluator disagrees with dense evaluator"
                );
            }
        }

        const std::string frequency_key = BuildFrequencyKey(move);
        const auto frequency_it = move_frequency_.find(frequency_key);
        const int frequency = frequency_it == move_frequency_.end()
            ? 0
            : frequency_it->second;
        const double diversity_weight = stagnation_count_ >= 4
            ? std::max(0.1, global_best_makespan_ * 0.0005)
            : 0.0;
        CandidateRecord record{
            std::move(schedule_items),
            move,
            std::move(signature),
            candidate_makespan,
            0.0
        };
        record.selection_score = record.makespan +
                                 diversity_weight * frequency;

        const bool aspiration = record.makespan < global_best_makespan_;
        if (!aspiration &&
            record.makespan >
                base_schedule_.get_TotalTime() + max_deterioration_) {
            return CanContinue();
        }
        if (!best_any_.has_value() ||
            record.makespan < best_any_->makespan ||
            (record.makespan == best_any_->makespan &&
             record.signature < best_any_->signature)) {
            best_any_ = record;
        }
        if (IsMoveTabu(move, tabu_expiration_, iteration_) && !aspiration) {
            return CanContinue();
        }

        if (!best_admissible_.has_value() ||
            IsBetterCandidate(record, *best_admissible_)) {
            best_admissible_ = record;
        }
        if (diversification_mode_) {
            diversification_pool_.push_back(std::move(record));
            std::sort(
                diversification_pool_.begin(),
                diversification_pool_.end(),
                IsBetterCandidate
            );
            if (static_cast<int>(diversification_pool_.size()) >
                kDiversificationPoolSize) {
                diversification_pool_.pop_back();
            }
        }
        return CanContinue();
    }

    bool CanContinue() const {
        return evaluated_count_ < kMaxNeighborEvaluations &&
               !SearchDeadlineReached(deadline_);
    }

    SelectionResult Select() {
        if (diversification_mode_ && !diversification_pool_.empty()) {
            const int selection_range = std::min(
                6,
                static_cast<int>(diversification_pool_.size())
            );
            const int selected_index = RandomInteger(0, selection_range - 1);
            return {diversification_pool_[selected_index], false};
        }
        if (best_admissible_.has_value()) {
            return {best_admissible_, false};
        }
        if (best_any_.has_value()) {
            return {best_any_, true};
        }
        return {};
    }

private:
    const Schedule &base_schedule_;
    const std::vector<Job> &jobs_;
    const std::vector<std::string> &job_list_;
    ScheduleItemsMakespanEvaluator sparse_evaluator_;
    std::vector<Machine> &machines_;
    const std::unordered_map<std::string, int> &tabu_expiration_;
    const std::unordered_map<std::string, int> &move_frequency_;
    int iteration_;
    int global_best_makespan_;
    int stagnation_count_;
    bool diversification_mode_;
    SearchDeadline deadline_;
    int max_deterioration_;
    int evaluated_count_ = 0;
    std::unordered_set<std::string> seen_signatures_;
    std::optional<CandidateRecord> best_admissible_;
    std::optional<CandidateRecord> best_any_;
    std::vector<CandidateRecord> diversification_pool_;
};

double BoundedRlRatio(const double numerator, const double denominator) {
    return std::clamp(
        numerator / std::max(1.0, denominator),
        -2.0,
        2.0
    );
}

class TabuRlCandidateCollector {
public:
    TabuRlCandidateCollector(
        const Schedule &base_schedule,
        const std::vector<Job> &jobs,
        const SearchDeadline deadline
    ) : base_schedule_(base_schedule),
        deadline_(deadline),
        original_items_(base_schedule.get_schedule_items()),
        original_locations_(BuildProcessLocationMap(original_items_)),
        machine_indices_(BuildMachineIndexMap(original_items_)),
        job_ids_(BuildJobIdMap(jobs)),
        workloads_(BuildMachineWorkloads(original_items_, job_ids_)),
        slack_(base_schedule.GetProcessSlack()),
        operation_count_(CountOperations(original_items_)) {
        candidates_.reserve(kMaxNeighborEvaluations);
        seen_signatures_.reserve(kMaxNeighborEvaluations * 2);
        seen_signatures_.insert(BuildScheduleSignature(original_items_));
    }

    bool Consider(
        std::vector<Schedule_item> &&schedule_items,
        const MoveAttribute &move,
        const TabuNeighborhoodType neighborhood
    ) {
        if (!CanContinue()) {
            return false;
        }
        std::string signature = BuildScheduleSignature(schedule_items);
        if (!seen_signatures_.insert(signature).second) {
            return true;
        }
        const GreedyRlFeatures features = BuildFeatures(
            schedule_items,
            move,
            neighborhood
        );
        candidates_.push_back({
            std::move(schedule_items),
            move,
            std::move(signature),
            features,
            neighborhood
        });
        return CanContinue();
    }

    bool CanContinue() const {
        return static_cast<int>(candidates_.size()) < kMaxNeighborEvaluations &&
               !SearchDeadlineReached(deadline_);
    }

    std::vector<TabuRlUnevaluatedCandidate> &Candidates() {
        return candidates_;
    }

private:
    GreedyRlFeatures BuildFeatures(
        const std::vector<Schedule_item> &candidate_items,
        const MoveAttribute &move,
        const TabuNeighborhoodType neighborhood
    ) const {
        GreedyRlFeatures features{};
        const int job_id = static_cast<int>(move.process_identifier >> 32);
        const int process_id = static_cast<int>(
            static_cast<unsigned int>(move.process_identifier)
        );
        const auto job_it = job_ids_.find(job_id);
        const auto source_machine_it = machine_indices_.find(move.source_machine_id);
        const auto target_machine_it = machine_indices_.find(move.target_machine_id);
        const auto source_location_it = original_locations_.find(
            move.process_identifier
        );
        if (job_it == job_ids_.end() ||
            source_machine_it == machine_indices_.end() ||
            target_machine_it == machine_indices_.end() ||
            source_location_it == original_locations_.end()) {
            return features;
        }
        const Job_process *job_process = FindJobProcess(
            *job_it->second,
            process_id
        );
        if (job_process == nullptr) {
            return features;
        }

        const int source_machine_index = source_machine_it->second;
        const int target_machine_index = target_machine_it->second;
        const auto &target_processes =
            candidate_items[target_machine_index].schedule_process;
        int target_position = -1;
        for (int position = 0;
             position < static_cast<int>(target_processes.size());
             ++position) {
            const auto &candidate_process = target_processes[position];
            if (BuildProcessId(
                    candidate_process.job_id,
                    candidate_process.process_id
                ) == move.process_identifier) {
                target_position = position;
                break;
            }
        }
        if (target_position < 0) {
            return features;
        }
        if (SparseEvaluatorVerificationEnabled()) {
            const auto candidate_locations =
                BuildProcessLocationMap(candidate_items);
            const auto target_location_it = candidate_locations.find(
                move.process_identifier
            );
            if (target_location_it == candidate_locations.end() ||
                target_location_it->second.machine_index !=
                    target_machine_index ||
                target_location_it->second.item_index != target_position) {
                throw std::runtime_error(
                    "optimized Tabu target lookup disagrees with full map"
                );
            }
        }
        const int source_length = static_cast<int>(
            original_items_[source_machine_index].schedule_process.size()
        );
        int target_length = static_cast<int>(
            candidate_items[target_machine_index].schedule_process.size()
        );
        const bool cross_machine = move.source_machine_id != move.target_machine_id;
        const bool swap = neighborhood == TabuNeighborhoodType::Compound;
        if (cross_machine) {
            target_length = std::max(0, target_length - 1);
        } else if (!swap) {
            target_length = std::max(0, source_length - 1);
        }

        const int source_processing_time = GetProcessingTime(
            *job_process,
            move.source_machine_id
        );
        const int target_processing_time = GetProcessingTime(
            *job_process,
            move.target_machine_id
        );
        const std::string process_key = BuildProcessKey(job_id, process_id);
        const auto slack_it = slack_.find(process_key);
        const int process_slack = slack_it == slack_.end() ? 0 : slack_it->second;
        int move_type = 0;
        if (swap) {
            move_type = 2;
        } else if (cross_machine) {
            move_type = 1;
        }

        features[0] = move_type == 0 ? 1.0 : 0.0;
        features[1] = move_type == 1 ? 1.0 : 0.0;
        features[2] = move_type == 2 ? 1.0 : 0.0;
        features[3] = process_slack == 0 ? 1.0 : 0.0;
        features[4] = BoundedRlRatio(
            process_slack,
            base_schedule_.get_TotalTime()
        );
        features[5] = BoundedRlRatio(
            source_processing_time,
            base_schedule_.get_TotalTime()
        );
        features[6] = BoundedRlRatio(
            target_processing_time,
            base_schedule_.get_TotalTime()
        );
        features[7] = BoundedRlRatio(
            source_processing_time - target_processing_time,
            base_schedule_.get_TotalTime()
        );
        features[8] = BoundedRlRatio(
            workloads_[source_machine_index],
            base_schedule_.get_TotalTime()
        );
        features[9] = BoundedRlRatio(
            workloads_[target_machine_index],
            base_schedule_.get_TotalTime()
        );
        features[10] = BoundedRlRatio(
            workloads_[source_machine_index] - workloads_[target_machine_index],
            base_schedule_.get_TotalTime()
        );
        features[11] = BoundedRlRatio(
            source_location_it->second.item_index,
            source_length - 1
        );
        features[12] = BoundedRlRatio(
            target_position,
            target_length
        );
        features[13] = BoundedRlRatio(
            std::abs(
                target_position -
                source_location_it->second.item_index
            ),
            std::max(source_length, target_length)
        );
        features[14] = BoundedRlRatio(source_length, operation_count_);
        features[15] = BoundedRlRatio(target_length, operation_count_);
        return features;
    }

    const Schedule &base_schedule_;
    SearchDeadline deadline_;
    const std::vector<Schedule_item> &original_items_;
    std::unordered_map<long long, ProcessLocation> original_locations_;
    std::unordered_map<int, int> machine_indices_;
    std::unordered_map<int, const Job *> job_ids_;
    std::vector<int> workloads_;
    std::unordered_map<std::string, int> slack_;
    int operation_count_;
    std::unordered_set<std::string> seen_signatures_;
    std::vector<TabuRlUnevaluatedCandidate> candidates_;
};

bool SubmitSameMachineMove(
    const std::vector<Schedule_item> &original_items,
    const int machine_index,
    const int source_position,
    const int insertion_position,
    const CandidateVisitor &visitor
) {
    if (source_position == insertion_position) {
        return true;
    }
    std::vector<Schedule_item> candidate_items = original_items;
    auto &processes = candidate_items[machine_index].schedule_process;
    const Schedule_process moved_process = processes[source_position];
    processes.erase(processes.begin() + source_position);
    processes.insert(processes.begin() + insertion_position, moved_process);
    candidate_items[machine_index].process_count =
        static_cast<int>(processes.size());

    const auto move = DescribeMove(original_items, candidate_items, moved_process);
    return !move.has_value() || visitor(std::move(candidate_items), *move);
}

void GenerateCriticalBlockNeighbors(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const CriticalProcesses &critical_processes,
    const CandidateVisitor &visitor
) {
    const auto &original_items = schedule.get_schedule_items();
    const auto blocks = BuildCriticalBlocks(schedule, jobs, critical_processes);
    int submitted_count = 0;

    for (const auto &block : blocks) {
        // Endpoint moves are the classic N5 neighborhood. Adjacent internal
        // swaps add low-cost alternatives for longer critical blocks.
        for (int position = block.first_position + 1;
             position < block.last_position;
             ++position) {
            if (submitted_count >= kMaxCriticalBlockEvaluations ||
                !SubmitSameMachineMove(
                    original_items,
                    block.machine_index,
                    position,
                    position + 1,
                    visitor
                )) {
                return;
            }
            ++submitted_count;
        }
        for (int insertion_position = block.first_position + 1;
             insertion_position <= block.last_position;
             ++insertion_position) {
            if (submitted_count >= kMaxCriticalBlockEvaluations ||
                !SubmitSameMachineMove(
                    original_items,
                    block.machine_index,
                    block.first_position,
                    insertion_position,
                    visitor
                )) {
                return;
            }
            ++submitted_count;
        }
        for (int insertion_position = block.first_position;
             insertion_position < block.last_position;
             ++insertion_position) {
            if (submitted_count >= kMaxCriticalBlockEvaluations ||
                !SubmitSameMachineMove(
                    original_items,
                    block.machine_index,
                    block.last_position,
                    insertion_position,
                    visitor
                )) {
                return;
            }
            ++submitted_count;
        }
    }
}

void GenerateMachineMoveNeighbors(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const CriticalProcesses &critical_processes,
    const CandidateVisitor &visitor
) {
    const auto &original_items = schedule.get_schedule_items();
    const auto location_map = BuildProcessLocationMap(original_items);
    const auto machine_index_map = BuildMachineIndexMap(original_items);
    const auto job_id_map = BuildJobIdMap(jobs);
    int submitted_count = 0;
    int critical_count = 0;

    for (const auto &critical_process : critical_processes.ordered) {
        if (critical_count++ >= kMaxCriticalProcesses) {
            return;
        }
        const auto location_it = location_map.find(critical_process.identifier);
        const auto job_it = job_id_map.find(critical_process.job_id);
        if (location_it == location_map.end() || job_it == job_id_map.end()) {
            continue;
        }
        const Job_process *job_process = FindJobProcess(
            *job_it->second,
            critical_process.process_id
        );
        if (job_process == nullptr) {
            continue;
        }

        const int source_machine_index = location_it->second.machine_index;
        const int source_position = location_it->second.item_index;
        const int source_machine_id = original_items[source_machine_index].machine_id;
        const Schedule_process moved_process =
            original_items[source_machine_index].schedule_process[source_position];
        std::unordered_set<int> visited_target_machines;
        std::vector<const Process_item *> machine_options;
        machine_options.reserve(job_process->process_item.size());
        for (const auto &machine_option : job_process->process_item) {
            machine_options.push_back(&machine_option);
        }
        std::stable_sort(
            machine_options.begin(),
            machine_options.end(),
            [](const Process_item *left, const Process_item *right) {
                return left->process_time < right->process_time;
            }
        );

        for (const Process_item *machine_option : machine_options) {
            const int target_machine_id = machine_option->machine_id;
            if (target_machine_id == source_machine_id ||
                !visited_target_machines.insert(target_machine_id).second) {
                continue;
            }
            const auto target_machine_it = machine_index_map.find(target_machine_id);
            if (target_machine_it == machine_index_map.end()) {
                continue;
            }
            const int target_machine_index = target_machine_it->second;
            const int target_process_count = static_cast<int>(
                original_items[target_machine_index].schedule_process.size()
            );

            for (int insertion_position = 0;
                 insertion_position <= target_process_count;
                 ++insertion_position) {
                if (submitted_count >= kMaxMachineMoveEvaluations) {
                    return;
                }
                std::vector<Schedule_item> candidate_items = original_items;
                auto &source_processes =
                    candidate_items[source_machine_index].schedule_process;
                source_processes.erase(source_processes.begin() + source_position);
                candidate_items[source_machine_index].process_count =
                    static_cast<int>(source_processes.size());

                auto &target_processes =
                    candidate_items[target_machine_index].schedule_process;
                target_processes.insert(
                    target_processes.begin() + insertion_position,
                    moved_process
                );
                candidate_items[target_machine_index].process_count =
                    static_cast<int>(target_processes.size());

                const auto move = DescribeMove(
                    original_items,
                    candidate_items,
                    moved_process
                );
                ++submitted_count;
                if (move.has_value() &&
                    !visitor(std::move(candidate_items), *move)) {
                    return;
                }
            }
        }
    }
}

void GenerateSequenceMoveNeighbors(
    const Schedule &schedule,
    const CriticalProcesses &critical_processes,
    const CandidateVisitor &visitor
) {
    const auto &original_items = schedule.get_schedule_items();
    const auto location_map = BuildProcessLocationMap(original_items);
    int submitted_count = 0;
    int critical_count = 0;

    for (const auto &critical_process : critical_processes.ordered) {
        if (critical_count++ >= kMaxCriticalProcesses) {
            return;
        }
        const auto location_it = location_map.find(critical_process.identifier);
        if (location_it == location_map.end()) {
            continue;
        }
        const int machine_index = location_it->second.machine_index;
        const int source_position = location_it->second.item_index;
        const int process_count = static_cast<int>(
            original_items[machine_index].schedule_process.size()
        );

        for (int insertion_position = 0;
             insertion_position < process_count;
             ++insertion_position) {
            if (insertion_position == source_position) {
                continue;
            }
            if (submitted_count >= kMaxSequenceMoveEvaluations ||
                !SubmitSameMachineMove(
                    original_items,
                    machine_index,
                    source_position,
                    insertion_position,
                    visitor
                )) {
                return;
            }
            ++submitted_count;
        }
    }
}

void GeneratePathRelinkingNeighbors(
    const Schedule &schedule,
    const Schedule &guide_schedule,
    const CandidateVisitor &visitor
) {
    const auto &original_items = schedule.get_schedule_items();
    const auto &guide_items = guide_schedule.get_schedule_items();
    if (original_items.size() != guide_items.size()) {
        return;
    }
    const auto current_location_map = BuildProcessLocationMap(original_items);
    int submitted_count = 0;
    constexpr int kMaxPathRelinkingMoves = 200;

    for (int target_machine_index = 0;
         target_machine_index < static_cast<int>(guide_items.size());
         ++target_machine_index) {
        const auto &target_processes =
            guide_items[target_machine_index].schedule_process;
        for (int target_position = 0;
             target_position < static_cast<int>(target_processes.size());
             ++target_position) {
            if (submitted_count >= kMaxPathRelinkingMoves) {
                return;
            }
            const Schedule_process moved_process = target_processes[target_position];
            const long long identifier = BuildProcessId(
                moved_process.job_id,
                moved_process.process_id
            );
            const auto current_it = current_location_map.find(identifier);
            if (current_it == current_location_map.end()) {
                continue;
            }
            if (current_it->second.machine_index == target_machine_index &&
                current_it->second.item_index == target_position) {
                continue;
            }

            std::vector<Schedule_item> candidate_items = original_items;
            auto &source_processes =
                candidate_items[current_it->second.machine_index].schedule_process;
            source_processes.erase(source_processes.begin() + current_it->second.item_index);
            candidate_items[current_it->second.machine_index].process_count =
                static_cast<int>(source_processes.size());
            auto &destination_processes =
                candidate_items[target_machine_index].schedule_process;
            const int insertion_position = std::min(
                target_position,
                static_cast<int>(destination_processes.size())
            );
            destination_processes.insert(
                destination_processes.begin() + insertion_position,
                moved_process
            );
            candidate_items[target_machine_index].process_count =
                static_cast<int>(destination_processes.size());
            const auto move = DescribeMove(
                original_items,
                candidate_items,
                moved_process
            );
            ++submitted_count;
            if (move.has_value() &&
                !visitor(std::move(candidate_items), *move)) {
                return;
            }
        }
    }
}

void GenerateCompoundNeighbors(
    const Schedule &schedule,
    const CriticalProcesses &critical_processes,
    const CandidateVisitor &visitor
) {
    const auto &original_items = schedule.get_schedule_items();
    const auto location_map = BuildProcessLocationMap(original_items);
    std::vector<std::pair<int, int>> locations;
    locations.reserve(critical_processes.ordered.size());
    for (const auto &process : critical_processes.ordered) {
        const auto location_it = location_map.find(process.identifier);
        if (location_it != location_map.end()) {
            locations.emplace_back(
                location_it->second.machine_index,
                location_it->second.item_index
            );
        }
    }

    int submitted_count = 0;
    constexpr int kMaxCompoundMoves = 250;
    for (int left = 0; left < static_cast<int>(locations.size()); ++left) {
        for (int right = left + 1;
             right < static_cast<int>(locations.size());
             ++right) {
            if (submitted_count >= kMaxCompoundMoves) {
                return;
            }
            if (locations[left].first != locations[right].first ||
                locations[left].second == locations[right].second) {
                continue;
            }
            std::vector<Schedule_item> candidate_items = original_items;
            auto &processes = candidate_items[locations[left].first].schedule_process;
            const Schedule_process moved_process = processes[locations[left].second];
            const Schedule_process second_moved_process =
                processes[locations[right].second];
            std::swap(
                processes[locations[left].second],
                processes[locations[right].second]
            );
            auto move = DescribeMove(
                original_items,
                candidate_items,
                moved_process
            );
            const auto second_move = DescribeMove(
                original_items,
                candidate_items,
                second_moved_process
            );
            if (move.has_value() && second_move.has_value()) {
                move->extra_destination_attributes =
                    BuildDestinationAttributes(*second_move);
                move->extra_reverse_attributes =
                    BuildReverseAttributes(*second_move);
            }
            ++submitted_count;
            if (move.has_value() &&
                !visitor(std::move(candidate_items), *move)) {
                return;
            }
        }
    }
}

void RemoveExpiredTabuAttributes(
    std::unordered_map<std::string, int> &tabu_expiration,
    const int iteration
) {
    for (auto it = tabu_expiration.begin(); it != tabu_expiration.end();) {
        if (it->second <= iteration) {
            it = tabu_expiration.erase(it);
        } else {
            ++it;
        }
    }
}

int GenerateTabuTenure(
    const int configured_tenure,
    const int operation_count,
    const int stagnation_count
) {
    if (configured_tenure <= 0) {
        return 0;
    }
    const int size_tenure = std::max(
        7,
        static_cast<int>(std::lround(std::sqrt(std::max(1, operation_count))))
    );
    const int center = std::max(5, (configured_tenure + size_tenure) / 2);
    const int lower = std::max(3, center * 2 / 3);
    int upper = std::max(lower, center * 4 / 3);
    if (stagnation_count >= 5) {
        upper += std::max(1, size_tenure / 2);
    }
    return RandomInteger(lower, upper);
}

void RegisterReverseMove(
    const MoveAttribute &move,
    const int expiration_iteration,
    std::unordered_map<std::string, int> &tabu_expiration
) {
    for (const auto &attribute : BuildReverseAttributes(move)) {
        auto [it, inserted] = tabu_expiration.emplace(
            attribute,
            expiration_iteration
        );
        if (!inserted) {
            it->second = std::max(it->second, expiration_iteration);
        }
    }
}

std::vector<Schedule> CollectNeighborhood(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    std::vector<Machine> &machines,
    const bool machine_moves
) {
    std::vector<Schedule> schedules;
    std::unordered_set<std::string> seen_signatures;
    seen_signatures.reserve(kMaxNeighborEvaluations * 2);
    seen_signatures.insert(BuildScheduleSignature(schedule.get_schedule_items()));
    int evaluated_count = 0;
    const CriticalProcesses critical_processes = BuildCriticalProcesses(
        schedule,
        false
    );

    const CandidateVisitor visitor = [&](
        std::vector<Schedule_item> &&schedule_items,
        const MoveAttribute &
    ) {
        if (evaluated_count >= kMaxNeighborEvaluations) {
            return false;
        }
        const std::string signature = BuildScheduleSignature(schedule_items);
        if (!seen_signatures.insert(signature).second) {
            return true;
        }
        ++evaluated_count;
        Schedule candidate;
        if (EvaluateCandidate(
                schedule,
                std::move(schedule_items),
                jobs,
                job_list,
                machines,
                candidate
            )) {
            schedules.push_back(std::move(candidate));
        }
        return evaluated_count < kMaxNeighborEvaluations;
    };

    if (machine_moves) {
        GenerateMachineMoveNeighbors(schedule, jobs, critical_processes, visitor);
    } else {
        GenerateCriticalBlockNeighbors(schedule, jobs, critical_processes, visitor);
        if (evaluated_count < kMaxNeighborEvaluations) {
            GenerateSequenceMoveNeighbors(schedule, critical_processes, visitor);
        }
    }
    return schedules;
}

}  // namespace

void SetTabuSearchSeed(const unsigned int random_seed) {
    GetTabuRng().seed(random_seed);
}

Schedule TabuSearch(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const int tabu_list_length,
    const int max_iter_count,
    std::vector<Machine> &machines,
    const SearchDeadline deadline,
    const bool use_persistent_state
) {
    if (max_iter_count <= 0 || schedule.get_schedule_items().empty() ||
        SearchDeadlineReached(deadline)) {
        return schedule;
    }

    PersistentTabuState isolated_state;
    PersistentTabuState &persistent_state = use_persistent_state
        ? GetPersistentTabuState()
        : isolated_state;
    const std::string problem_signature = BuildProblemSignature(jobs, jobList);
    if (persistent_state.problem_signature != problem_signature) {
        persistent_state = PersistentTabuState{};
        persistent_state.problem_signature = problem_signature;
    }

    Schedule current_schedule = schedule;
    const int persistent_tolerance = std::max(
        5,
        static_cast<int>(std::lround(schedule.get_TotalTime() * 0.05))
    );
    bool resumed_persistent_trajectory = false;
    if (persistent_state.current_schedule.has_value() &&
        persistent_state.current_schedule->get_TotalTime() <=
            schedule.get_TotalTime() + persistent_tolerance) {
        current_schedule = *persistent_state.current_schedule;
        resumed_persistent_trajectory = true;
    }
    if (!resumed_persistent_trajectory) {
        persistent_state.tabu_expiration.clear();
    }
    Schedule global_best_schedule = schedule;
    int global_best_makespan = schedule.get_TotalTime();
    if (current_schedule.get_TotalTime() < global_best_makespan) {
        global_best_schedule = current_schedule;
        global_best_makespan = current_schedule.get_TotalTime();
    }
    const int operation_count = CountOperations(schedule.get_schedule_items());
    const int default_stagnation_limit = std::max(
        8,
        std::min(40, std::max(1, max_iter_count / 3))
    );
    const int stagnation_limit = EnvironmentInteger(
        "HLS_TABU_STAGNATION_LIMIT",
        default_stagnation_limit,
        4,
        200
    );
    const int perturbation_min = EnvironmentInteger(
        "HLS_TABU_PERTURB_MIN",
        2,
        1,
        100
    );
    const int perturbation_max = std::max(
        perturbation_min,
        EnvironmentInteger(
            "HLS_TABU_PERTURB_MAX",
            5,
            1,
            100
        )
    );
    int stagnation_count = 0;
    int perturbation_steps_remaining = 0;
    std::unordered_map<std::string, int> &tabu_expiration =
        persistent_state.tabu_expiration;
    std::unordered_map<std::string, int> &move_frequency =
        persistent_state.move_frequency;
    tabu_expiration.reserve(std::max(32, tabu_list_length * 4));
    move_frequency.reserve(std::max(64, operation_count * 4));

    int iterations_completed = 0;
    for (int iteration = 0; iteration < max_iter_count; ++iteration) {
        if (SearchDeadlineReached(deadline)) {
            break;
        }
        const int absolute_iteration =
            persistent_state.absolute_iteration + iteration;
        RemoveExpiredTabuAttributes(tabu_expiration, absolute_iteration);
        CriticalProcesses critical_processes = BuildCriticalProcesses(
            current_schedule,
            stagnation_count >= 6 || perturbation_steps_remaining > 0
        );
        RotateCriticalCandidates(critical_processes, iteration);
        if (critical_processes.ordered.empty()) {
            break;
        }

        NeighborEvaluator evaluator(
            current_schedule,
            jobs,
            jobList,
            machines,
            tabu_expiration,
            move_frequency,
            absolute_iteration,
            global_best_makespan,
            stagnation_count,
            perturbation_steps_remaining > 0,
            deadline,
            std::max(
                3,
                static_cast<int>(std::lround(global_best_makespan * 0.02)) +
                    std::min(20, stagnation_count * 2)
            )
        );
        SelectionResult selection;
        if (!TabuRlPolicyEnabled()) {
            const CandidateVisitor visitor = [&](
                std::vector<Schedule_item> &&schedule_items,
                const MoveAttribute &move
            ) {
                return evaluator.Consider(std::move(schedule_items), move);
            };

            GenerateCriticalBlockNeighbors(
                current_schedule,
                jobs,
                critical_processes,
                visitor
            );
            if (evaluator.CanContinue() &&
                (stagnation_count >= 4 || perturbation_steps_remaining > 0) &&
                ScheduleToString(current_schedule) !=
                    ScheduleToString(global_best_schedule)) {
                GeneratePathRelinkingNeighbors(
                    current_schedule,
                    global_best_schedule,
                    visitor
                );
            }
            if (evaluator.CanContinue()) {
                GenerateMachineMoveNeighbors(
                    current_schedule,
                    jobs,
                    critical_processes,
                    visitor
                );
            }
            if (evaluator.CanContinue() && stagnation_count >= 4) {
                GenerateCompoundNeighbors(
                    current_schedule,
                    critical_processes,
                    visitor
                );
            }
            if (evaluator.CanContinue()) {
                GenerateSequenceMoveNeighbors(
                    current_schedule,
                    critical_processes,
                    visitor
                );
            }
            selection = evaluator.Select();
        } else {
            TabuRlCandidateCollector collector(
                current_schedule,
                jobs,
                deadline
            );
            auto collector_visitor = [&](const TabuNeighborhoodType type) {
                return CandidateVisitor([
                    &collector,
                    type
                ](
                    std::vector<Schedule_item> &&schedule_items,
                    const MoveAttribute &move
                ) {
                    return collector.Consider(
                        std::move(schedule_items),
                        move,
                        type
                    );
                });
            };

            GenerateCriticalBlockNeighbors(
                current_schedule,
                jobs,
                critical_processes,
                collector_visitor(TabuNeighborhoodType::CriticalBlock)
            );
            if (collector.CanContinue() &&
                (stagnation_count >= 4 || perturbation_steps_remaining > 0) &&
                ScheduleToString(current_schedule) !=
                    ScheduleToString(global_best_schedule)) {
                GeneratePathRelinkingNeighbors(
                    current_schedule,
                    global_best_schedule,
                    collector_visitor(TabuNeighborhoodType::PathRelinking)
                );
            }
            if (collector.CanContinue()) {
                GenerateMachineMoveNeighbors(
                    current_schedule,
                    jobs,
                    critical_processes,
                    collector_visitor(TabuNeighborhoodType::MachineMove)
                );
            }
            if (collector.CanContinue() && stagnation_count >= 4) {
                GenerateCompoundNeighbors(
                    current_schedule,
                    critical_processes,
                    collector_visitor(TabuNeighborhoodType::Compound)
                );
            }
            if (collector.CanContinue()) {
                GenerateSequenceMoveNeighbors(
                    current_schedule,
                    critical_processes,
                    collector_visitor(TabuNeighborhoodType::SequenceMove)
                );
            }

            auto &candidates = collector.Candidates();
            std::vector<GreedyRlFeatures> features;
            features.reserve(candidates.size());
            for (const auto &candidate : candidates) {
                features.push_back(candidate.features);
            }
            std::vector<std::size_t> selected_indices =
                SelectTabuRlCandidates(
                    features,
                    kMaxNeighborEvaluations,
                    operation_count
                );
            std::vector<bool> selected_mask(candidates.size(), false);
            for (const std::size_t index : selected_indices) {
                if (index < selected_mask.size()) {
                    selected_mask[index] = true;
                }
            }

            long long safeguard_count = 0;
            const int minimum_per_neighborhood =
                GetTabuRlRuntimeConfig().minimum_per_neighborhood;
            for (int type_index = 0;
                 type_index < kTabuNeighborhoodTypeCount;
                 ++type_index) {
                int available = 0;
                int already_selected = 0;
                for (std::size_t index = 0; index < candidates.size(); ++index) {
                    if (static_cast<int>(candidates[index].neighborhood) !=
                        type_index) {
                        continue;
                    }
                    ++available;
                    already_selected += selected_mask[index] ? 1 : 0;
                }
                const int target = std::min(
                    minimum_per_neighborhood,
                    available
                );
                for (std::size_t index = 0;
                     index < candidates.size() && already_selected < target;
                     ++index) {
                    if (static_cast<int>(candidates[index].neighborhood) ==
                            type_index &&
                        !selected_mask[index]) {
                        selected_mask[index] = true;
                        selected_indices.push_back(index);
                        ++already_selected;
                        ++safeguard_count;
                    }
                }
            }
            RecordTabuRlNeighborhoodSafeguards(safeguard_count);

            for (const std::size_t index : selected_indices) {
                if (index >= candidates.size() || !evaluator.CanContinue()) {
                    break;
                }
                evaluator.Consider(
                    std::move(candidates[index].schedule_items),
                    candidates[index].move,
                    std::move(candidates[index].signature)
                );
            }
            selection = evaluator.Select();
            const bool missing_candidate = !selection.candidate.has_value();
            const bool non_improving_candidate =
                selection.candidate.has_value() &&
                selection.candidate->makespan >=
                    current_schedule.get_TotalTime();
            const bool missed_global_improvement =
                selection.candidate.has_value() &&
                selection.candidate->makespan >= global_best_makespan &&
                selected_indices.size() < candidates.size();
            const bool verify_full_neighborhood =
                GetTabuRlRuntimeConfig().verify_full_neighborhood;
            const bool should_fallback = evaluator.CanContinue() && (
                verify_full_neighborhood ||
                missed_global_improvement ||
                (missing_candidate &&
                 GetTabuRlRuntimeConfig().fallback_on_miss) ||
                (non_improving_candidate &&
                 GetTabuRlRuntimeConfig().fallback_on_non_improvement)
            );
            if (should_fallback) {
                if (verify_full_neighborhood) {
                    RecordTabuRlVerification();
                } else if (missing_candidate) {
                    RecordTabuRlFallback();
                } else if (missed_global_improvement) {
                    RecordTabuRlIntensificationExpansion();
                } else {
                    RecordTabuRlNonImprovementFallback();
                }
                for (std::size_t index = 0;
                     index < candidates.size() && evaluator.CanContinue();
                     ++index) {
                    if (selected_mask[index]) {
                        continue;
                    }
                    evaluator.Consider(
                        std::move(candidates[index].schedule_items),
                        candidates[index].move,
                        std::move(candidates[index].signature)
                    );
                }
                selection = evaluator.Select();
            }
        }

        if (!selection.candidate.has_value()) {
            break;
        }
        if (selection.forced_tabu_move) {
            tabu_expiration.clear();
        }

        CandidateRecord selected = std::move(*selection.candidate);
        Schedule selected_schedule;
        if (!EvaluateCandidate(
                current_schedule,
                std::move(selected.schedule_items),
                jobs,
                jobList,
                machines,
                selected_schedule
            ) ||
            selected_schedule.get_TotalTime() != selected.makespan) {
            throw std::runtime_error(
                "selected Tabu candidate disagrees with sparse evaluation"
            );
        }
        const int tenure = GenerateTabuTenure(
            tabu_list_length,
            operation_count,
            stagnation_count
        );
        if (tenure > 0) {
            RegisterReverseMove(
                selected.move,
                absolute_iteration + tenure + 1,
                tabu_expiration
            );
        }
        ++move_frequency[BuildFrequencyKey(selected.move)];
        current_schedule = std::move(selected_schedule);
        ++iterations_completed;

        if (current_schedule.get_TotalTime() < global_best_makespan) {
            global_best_makespan = current_schedule.get_TotalTime();
            global_best_schedule = current_schedule;
            stagnation_count = 0;
            perturbation_steps_remaining = 0;
        } else {
            ++stagnation_count;
            if (perturbation_steps_remaining > 0) {
                --perturbation_steps_remaining;
            }
        }

        if (stagnation_count >= stagnation_limit &&
            iteration + 1 < max_iter_count) {
            current_schedule = global_best_schedule;
            tabu_expiration.clear();
            perturbation_steps_remaining = RandomInteger(
                perturbation_min,
                perturbation_max
            );
            stagnation_count = 0;
        }
    }

    persistent_state.absolute_iteration += iterations_completed;
    persistent_state.current_schedule = current_schedule;
    if (persistent_state.absolute_iteration > 0 &&
        persistent_state.absolute_iteration % 300 < iterations_completed) {
        for (auto it = move_frequency.begin(); it != move_frequency.end();) {
            it->second /= 2;
            if (it->second == 0) {
                it = move_frequency.erase(it);
            } else {
                ++it;
            }
        }
    }

    CalculateTotalFailureRate(global_best_schedule, machines);
    return global_best_schedule;
}

std::vector<Schedule> MoveProcessSearch(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    std::vector<Machine> &machines
) {
    return CollectNeighborhood(
        schedule,
        jobs,
        jobList,
        machines,
        true
    );
}

std::vector<Schedule> ExchangeProcessSearch(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    std::vector<Machine> &machines
) {
    return CollectNeighborhood(
        schedule,
        jobs,
        jobList,
        machines,
        false
    );
}

std::string ScheduleToString(const Schedule &schedule) {
    return BuildScheduleSignature(schedule.get_schedule_items());
}

bool IsInTabuList(
    const Schedule &schedule,
    const std::unordered_set<std::string> &tabu_set
) {
    return tabu_set.contains(ScheduleToString(schedule));
}

void AddScheduleToTabuList(
    const Schedule &schedule,
    std::deque<std::string> &tabu_queue,
    std::unordered_set<std::string> &tabu_set,
    const int tabu_list_length
) {
    if (tabu_list_length <= 0) {
        return;
    }
    const std::string schedule_string = ScheduleToString(schedule);
    if (tabu_set.contains(schedule_string)) {
        return;
    }
    if (static_cast<int>(tabu_queue.size()) >= tabu_list_length) {
        tabu_set.erase(tabu_queue.front());
        tabu_queue.pop_front();
    }
    tabu_queue.push_back(schedule_string);
    tabu_set.insert(schedule_string);
}
