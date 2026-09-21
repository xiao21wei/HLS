//
// Created by 28898 on 25-1-3.
//

#include "RandomSearch.h"

#include <algorithm>
#include <array>
#include <climits>
#include <ctime>
#include <iostream>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataProc.h"
#include "GreedySearch.h"
#include "TabuSearch.h"

namespace {

constexpr double kCrossoverRate = 0.85;
constexpr double kBaseMutationRate = 0.12;
constexpr double kElevatedMutationRate = 0.25;
constexpr double kHighMutationRate = 0.45;
constexpr int kElevatedMutationStagnation = 6;
constexpr int kHighMutationStagnation = 12;
constexpr int kTournamentSize = 3;
constexpr int kLocalSearchInterval = 10;
constexpr double kMinimumPopulationDistance = 0.06;
constexpr double kPopulationRestartDiversity = 0.08;

using JobNameMap = std::unordered_map<std::string, const Job *>;

struct Individual {
    std::vector<int> machine_code;
    std::vector<int> operation_code;
    int fitness = INT_MAX;
};

struct GeneticOperatorStats {
    int attempts = 0;
    int improvements = 0;
    double total_gain = 0.0;
};

struct PersistentGeneticState {
    std::string problem_signature;
    std::vector<Individual> population;
    std::unordered_map<std::string, int> fitness_cache;
    std::array<GeneticOperatorStats, 2> operator_stats{};
    int generation = 0;
    int stagnation_count = 0;
    int calls_without_population_improvement = 0;
};

PersistentGeneticState &GetPersistentGeneticState() {
    static thread_local PersistentGeneticState state;
    return state;
}

std::mt19937 &GetSharedRng() {
    static thread_local std::mt19937 rng(5489U);
    return rng;
}

double RandomProbability() {
    return std::generate_canonical<double, 53>(GetSharedRng());
}

bool RandomChance(const double probability) {
    return RandomProbability() < probability;
}

int RandomIndex(const int count) {
    if (count <= 0) {
        throw std::invalid_argument("RandomIndex requires a positive count.");
    }
    return std::uniform_int_distribution<int>(0, count - 1)(GetSharedRng());
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

JobNameMap BuildJobNameMap(const std::vector<Job> &jobs) {
    JobNameMap job_name_map;
    job_name_map.reserve(jobs.size());
    for (const auto &job : jobs) {
        job_name_map[job.get_job_name()] = &job;
    }
    return job_name_map;
}

std::unordered_map<int, const Job *> BuildJobIdMap(const std::vector<Job> &jobs) {
    std::unordered_map<int, const Job *> job_id_map;
    job_id_map.reserve(jobs.size());
    for (const auto &job : jobs) {
        job_id_map[job.get_job_id()] = &job;
    }
    return job_id_map;
}

std::vector<const Job *> BuildOrderedJobsByJobList(
    const std::vector<std::string> &jobList,
    const JobNameMap &job_name_map
) {
    std::vector<const Job *> ordered_jobs;
    ordered_jobs.reserve(jobList.size());
    for (const auto &job_name : jobList) {
        const auto job_it = job_name_map.find(job_name);
        ordered_jobs.push_back(job_it == job_name_map.end() ? nullptr : job_it->second);
    }
    return ordered_jobs;
}

std::vector<const Job_process *> BuildOrderedProcesses(
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList
) {
    const auto job_name_map = BuildJobNameMap(jobs);
    const auto ordered_jobs = BuildOrderedJobsByJobList(jobList, job_name_map);
    std::vector<const Job_process *> ordered_processes;
    for (const Job *job : ordered_jobs) {
        if (job == nullptr) {
            continue;
        }
        for (const auto &job_process : job->get_job_process()) {
            ordered_processes.push_back(&job_process);
        }
    }
    return ordered_processes;
}

std::unordered_map<long long, int> BuildScheduleMachineMap(const Schedule &schedule) {
    std::unordered_map<long long, int> machine_map;
    for (const auto &schedule_item : schedule.get_schedule_items()) {
        for (const auto &process : schedule_item.schedule_process) {
            machine_map[BuildProcessId(process.job_id, process.process_id)] =
                schedule_item.machine_id;
        }
    }
    return machine_map;
}

std::string BuildChromosomeKey(
    const std::vector<int> &machine_selection_code,
    const std::vector<int> &operation_sequencing_code
) {
    std::string key;
    key.reserve((machine_selection_code.size() + operation_sequencing_code.size()) * 4 + 1);
    for (const int value : machine_selection_code) {
        key += std::to_string(value);
        key.push_back(',');
    }
    key.push_back('|');
    for (const int value : operation_sequencing_code) {
        key += std::to_string(value);
        key.push_back(',');
    }
    return key;
}

std::string BuildChromosomeKey(const Individual &individual) {
    return BuildChromosomeKey(individual.machine_code, individual.operation_code);
}

double ChromosomeDistance(
    const Individual &left,
    const Individual &right
) {
    int differences = 0;
    int compared = 0;
    const int machine_count = std::min(
        left.machine_code.size(),
        right.machine_code.size()
    );
    for (int index = 0; index < machine_count; ++index) {
        differences += left.machine_code[index] != right.machine_code[index];
    }
    compared += machine_count;
    const int operation_count = std::min(
        left.operation_code.size(),
        right.operation_code.size()
    );
    for (int index = 0; index < operation_count; ++index) {
        differences += left.operation_code[index] != right.operation_code[index];
    }
    compared += operation_count;
    return compared == 0
        ? 0.0
        : static_cast<double>(differences) / compared;
}

bool IsStructurallyDiverse(
    const Individual &candidate,
    const std::vector<Individual> &population,
    const double minimum_distance
) {
    return std::all_of(
        population.begin(),
        population.end(),
        [&](const Individual &individual) {
            return ChromosomeDistance(candidate, individual) >= minimum_distance;
        }
    );
}

double PopulationDiversity(const std::vector<Individual> &population) {
    if (population.size() <= 1) {
        return 0.0;
    }
    const Individual &best = *std::min_element(
        population.begin(),
        population.end(),
        [](const Individual &left, const Individual &right) {
            return left.fitness < right.fitness;
        }
    );
    double total_distance = 0.0;
    for (const auto &individual : population) {
        total_distance += ChromosomeDistance(best, individual);
    }
    return total_distance / population.size();
}

std::string BuildProblemSignature(
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList
) {
    std::string signature;
    signature.reserve(jobs.size() * 32 + jobList.size() * 12);
    for (const auto &job_name : jobList) {
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

std::vector<int> BuildTopologicalOperationCode(const Schedule &schedule) {
    const auto &graph = schedule.get_graph();
    const auto &process_list = schedule.get_processList();
    const auto &start_times = schedule.get_start_time();
    const int node_count = static_cast<int>(graph.size());
    if (node_count == 0 || process_list.size() != graph.size()) {
        return {};
    }

    std::vector<int> in_degree(node_count, 0);
    for (int from = 0; from < node_count; ++from) {
        for (int to = 0; to < node_count; ++to) {
            if (graph[from][to] != -1) {
                ++in_degree[to];
            }
        }
    }

    auto compare_nodes = [&start_times](const int left, const int right) {
        const int left_start = left < static_cast<int>(start_times.size())
            ? start_times[left]
            : 0;
        const int right_start = right < static_cast<int>(start_times.size())
            ? start_times[right]
            : 0;
        if (left_start != right_start) {
            return left_start > right_start;
        }
        return left > right;
    };
    std::priority_queue<int, std::vector<int>, decltype(compare_nodes)> ready(compare_nodes);
    for (int node = 0; node < node_count; ++node) {
        if (in_degree[node] == 0) {
            ready.push(node);
        }
    }

    std::vector<int> operation_code;
    operation_code.reserve(node_count > 2 ? node_count - 2 : 0);
    int visited_count = 0;
    while (!ready.empty()) {
        const int node = ready.top();
        ready.pop();
        ++visited_count;

        const auto parsed_process = ParseProcessKey(process_list[node]);
        if (parsed_process.has_value()) {
            operation_code.push_back(parsed_process->first);
        }
        for (int next = 0; next < node_count; ++next) {
            if (graph[node][next] != -1 && --in_degree[next] == 0) {
                ready.push(next);
            }
        }
    }

    if (visited_count != node_count) {
        return {};
    }
    return operation_code;
}

void EvaluateIndividual(
    Individual &individual,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines,
    std::unordered_map<std::string, int> &fitness_cache
) {
    individual.fitness = CalculateFitness(
        individual.machine_code,
        individual.operation_code,
        jobs,
        jobList,
        base_schedule,
        machines,
        fitness_cache
    );
}

bool AddUniqueIndividual(
    Individual individual,
    std::vector<Individual> &population,
    std::unordered_set<std::string> &population_keys,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines,
    std::unordered_map<std::string, int> &fitness_cache
) {
    const std::string key = BuildChromosomeKey(individual);
    if (!population_keys.insert(key).second) {
        return false;
    }
    EvaluateIndividual(
        individual,
        jobs,
        jobList,
        base_schedule,
        machines,
        fitness_cache
    );
    population.push_back(std::move(individual));
    return true;
}

void AppendGeneratedPopulation(
    const std::vector<std::vector<int>> &machine_codes,
    const std::vector<std::vector<int>> &operation_codes,
    const int target_size,
    std::vector<Individual> &population,
    std::unordered_set<std::string> &population_keys,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines,
    std::unordered_map<std::string, int> &fitness_cache,
    const SearchDeadline deadline
) {
    const int count = std::min(machine_codes.size(), operation_codes.size());
    for (int index = 0;
         index < count && static_cast<int>(population.size()) < target_size &&
             !SearchDeadlineReached(deadline);
         ++index) {
        AddUniqueIndividual(
            {machine_codes[index], operation_codes[index], INT_MAX},
            population,
            population_keys,
            jobs,
            jobList,
            base_schedule,
            machines,
            fitness_cache
        );
    }
}

std::vector<Individual> BuildInitialPopulation(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    const std::vector<std::string> &jobList,
    const int population_size,
    std::unordered_map<std::string, int> &fitness_cache,
    const SearchDeadline deadline
) {
    std::vector<int> incumbent_machine_code;
    std::vector<int> incumbent_operation_code;
    EncodeSchedule(
        schedule,
        jobs,
        jobList,
        incumbent_machine_code,
        incumbent_operation_code
    );

    std::vector<Individual> population;
    population.reserve(population_size);
    std::unordered_set<std::string> population_keys;
    population_keys.reserve(population_size * 2);
    AddUniqueIndividual(
        {incumbent_machine_code, incumbent_operation_code, INT_MAX},
        population,
        population_keys,
        jobs,
        jobList,
        schedule,
        machines,
        fitness_cache
    );

    const int perturbation_target = std::max(1, population_size / 5);
    for (int index = 0;
         index < perturbation_target && static_cast<int>(population.size()) < population_size;
         ++index) {
        if (SearchDeadlineReached(deadline)) {
            break;
        }
        Individual perturbed{incumbent_machine_code, incumbent_operation_code, INT_MAX};
        if (index % 2 == 0) {
            MachineVariation(perturbed.machine_code, jobs, jobList, perturbed.machine_code);
        } else {
            OperationVariation(perturbed.operation_code, perturbed.operation_code);
        }
        if (index % 3 == 2) {
            MachineVariation(perturbed.machine_code, jobs, jobList, perturbed.machine_code);
            OperationVariation(perturbed.operation_code, perturbed.operation_code);
        }
        AddUniqueIndividual(
            std::move(perturbed),
            population,
            population_keys,
            jobs,
            jobList,
            schedule,
            machines,
            fitness_cache
        );
    }

    std::vector<std::vector<int>> machine_codes;
    std::vector<std::vector<int>> operation_codes;
    const int global_count = std::max(1, population_size * 4 / 10);
    if (!SearchDeadlineReached(deadline)) {
        GlobalInitializePopulation(
            jobs,
            machines,
            jobList,
            global_count,
            incumbent_operation_code,
            machine_codes,
            operation_codes
        );
        AppendGeneratedPopulation(
            machine_codes,
            operation_codes,
            population_size,
            population,
            population_keys,
            jobs,
            jobList,
            schedule,
            machines,
            fitness_cache,
            deadline
        );
    }

    const int local_count = std::max(1, population_size / 5);
    if (!SearchDeadlineReached(deadline)) {
        LocalInitializePopulation(
            jobs,
            machines,
            jobList,
            local_count,
            incumbent_operation_code,
            machine_codes,
            operation_codes
        );
        AppendGeneratedPopulation(
            machine_codes,
            operation_codes,
            population_size,
            population,
            population_keys,
            jobs,
            jobList,
            schedule,
            machines,
            fitness_cache,
            deadline
        );
    }

    int attempts = 0;
    while (static_cast<int>(population.size()) < population_size &&
           attempts < population_size * 10 &&
           !SearchDeadlineReached(deadline)) {
        RandomInitializePopulation(
            jobs,
            machines,
            jobList,
            1,
            incumbent_operation_code,
            machine_codes,
            operation_codes
        );
        AppendGeneratedPopulation(
            machine_codes,
            operation_codes,
            population_size,
            population,
            population_keys,
            jobs,
            jobList,
            schedule,
            machines,
            fitness_cache,
            deadline
        );
        ++attempts;
    }

    int fallback_attempts = 0;
    while (static_cast<int>(population.size()) < population_size &&
           fallback_attempts < population_size * 20 &&
           !SearchDeadlineReached(deadline)) {
        Individual fallback = population.front();
        if (fallback_attempts % 2 == 0) {
            MachineVariation(
                fallback.machine_code,
                jobs,
                jobList,
                fallback.machine_code
            );
        } else {
            OperationVariation(
                fallback.operation_code,
                fallback.operation_code
            );
        }
        AddUniqueIndividual(
            std::move(fallback),
            population,
            population_keys,
            jobs,
            jobList,
            schedule,
            machines,
            fitness_cache
        );
        ++fallback_attempts;
    }
    // A deadline can make it impossible to generate a full unique population.
    // Keep the search valid, but only duplicate as a final fallback.
    while (static_cast<int>(population.size()) < population_size &&
           !population.empty()) {
        population.push_back(population.front());
    }
    return population;
}

int TournamentSelect(const std::vector<Individual> &population, const int excluded_index = -1) {
    int best_index = -1;
    for (int attempt = 0; attempt < kTournamentSize; ++attempt) {
        int candidate_index = RandomIndex(static_cast<int>(population.size()));
        if (population.size() > 1 && candidate_index == excluded_index) {
            candidate_index = (candidate_index + 1) % population.size();
        }
        if (best_index < 0 ||
            population[candidate_index].fitness < population[best_index].fitness) {
            best_index = candidate_index;
        }
    }
    return best_index;
}

bool ApplyCriticalVariation(
    Individual &individual,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines
);

bool MutateIndividual(
    Individual &individual,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const double mutation_rate,
    const Schedule &base_schedule,
    std::vector<Machine> &machines
) {
    if (!RandomChance(mutation_rate)) {
        return false;
    }
    bool critical_variation_applied = false;
    if (RandomChance(0.65)) {
        critical_variation_applied = ApplyCriticalVariation(
            individual,
            jobs,
            jobList,
            base_schedule,
            machines
        );
    }
    if (!critical_variation_applied && RandomChance(0.5)) {
        MachineVariation(
            individual.machine_code,
            jobs,
            jobList,
            individual.machine_code
        );
    } else if (!critical_variation_applied) {
        OperationVariation(individual.operation_code, individual.operation_code);
    }

    if (mutation_rate >= kHighMutationRate && RandomChance(0.25)) {
        MachineVariation(
            individual.machine_code,
            jobs,
            jobList,
            individual.machine_code
        );
        OperationVariation(individual.operation_code, individual.operation_code);
        critical_variation_applied = true;
    }
    return true;
}

bool ApplyCriticalVariation(
    Individual &individual,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines
) {
    Schedule decoded;
    try {
        decoded = DecodeSchedule(
            individual.machine_code,
            individual.operation_code,
            jobs,
            jobList,
            base_schedule,
            machines
        );
    } catch (const std::exception &) {
        return false;
    }

    const auto job_name_map = BuildJobNameMap(jobs);
    const auto ordered_jobs = BuildOrderedJobsByJobList(jobList, job_name_map);
    std::unordered_map<long long, int> machine_gene_map;
    int machine_gene = 0;
    for (const Job *job : ordered_jobs) {
        if (job == nullptr) {
            continue;
        }
        for (const auto &job_process : job->get_job_process()) {
            if (!job_process.process_item.empty()) {
                machine_gene_map[BuildProcessId(
                    job->get_job_id(),
                    job_process.process_item.front().process_id
                )] = machine_gene;
            }
            ++machine_gene;
        }
    }

    std::unordered_map<long long, int> operation_position_map;
    std::unordered_map<int, int> occurrence_count;
    for (int position = 0;
         position < static_cast<int>(individual.operation_code.size());
         ++position) {
        const int job_id = individual.operation_code[position];
        const int process_id = occurrence_count[job_id]++;
        operation_position_map[BuildProcessId(job_id, process_id)] = position;
    }

    std::vector<long long> critical_ids;
    for (const auto &process_key : decoded.GetKeyProcess()) {
        const auto parsed_process = ParseProcessKey(process_key);
        if (parsed_process.has_value()) {
            critical_ids.push_back(BuildProcessId(
                parsed_process->first,
                parsed_process->second
            ));
        }
    }
    if (critical_ids.empty()) {
        return false;
    }
    std::shuffle(critical_ids.begin(), critical_ids.end(), GetSharedRng());

    bool changed = false;
    const auto job_id_map = BuildJobIdMap(jobs);
    for (const long long identifier : critical_ids) {
        const int job_id = static_cast<int>(identifier >> 32);
        const int process_id = static_cast<int>(identifier & 0xffffffffU);
        const auto job_it = job_id_map.find(job_id);
        if (job_it == job_id_map.end()) {
            continue;
        }
        const Job_process *job_process = nullptr;
        for (const auto &candidate : job_it->second->get_job_process()) {
            if (!candidate.process_item.empty() &&
                candidate.process_item.front().process_id == process_id) {
                job_process = &candidate;
                break;
            }
        }
        const auto machine_gene_it = machine_gene_map.find(identifier);
        if (job_process != nullptr &&
            machine_gene_it != machine_gene_map.end() &&
            job_process->machine_count > 1) {
            std::vector<int> alternatives;
            for (int option = 0;
                 option < job_process->machine_count;
                 ++option) {
                if (option != individual.machine_code[machine_gene_it->second]) {
                    alternatives.push_back(option);
                }
            }
            std::sort(
                alternatives.begin(),
                alternatives.end(),
                [job_process](const int left, const int right) {
                    return job_process->process_item[left].process_time <
                           job_process->process_item[right].process_time;
                }
            );
            const int candidate_count = std::min(2, static_cast<int>(alternatives.size()));
            individual.machine_code[machine_gene_it->second] =
                alternatives[RandomIndex(candidate_count)];
            changed = true;
            break;
        }

        const auto operation_it = operation_position_map.find(identifier);
        if (operation_it == operation_position_map.end()) {
            continue;
        }
        const int source_position = operation_it->second;
        std::vector<int> other_positions;
        for (const auto &other : critical_ids) {
            const auto other_it = operation_position_map.find(other);
            if (other_it != operation_position_map.end() &&
                other_it->second != source_position &&
                individual.operation_code[other_it->second] !=
                    individual.operation_code[source_position]) {
                other_positions.push_back(other_it->second);
            }
        }
        if (!other_positions.empty()) {
            std::swap(
                individual.operation_code[source_position],
                individual.operation_code[other_positions[RandomIndex(
                    static_cast<int>(other_positions.size())
                )]]
            );
            changed = true;
            break;
        }
    }
    return changed;
}

std::vector<Individual> BuildOffspring(
    const std::vector<Individual> &population,
    const int offspring_count,
    const double mutation_rate,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines,
    std::unordered_map<std::string, int> &fitness_cache,
    const SearchDeadline deadline,
    std::array<GeneticOperatorStats, 2> &operator_stats
) {
    std::vector<Individual> offspring;
    offspring.reserve(offspring_count + 1);
    while (static_cast<int>(offspring.size()) < offspring_count &&
           !SearchDeadlineReached(deadline)) {
        const int first_parent_index = TournamentSelect(population);
        const int second_parent_index = TournamentSelect(population, first_parent_index);
        const Individual &first_parent = population[first_parent_index];
        const Individual &second_parent = population[second_parent_index];
        Individual first_child = first_parent;
        Individual second_child = second_parent;

        const double crossover_rate = operator_stats[0].attempts < 8
            ? kCrossoverRate
            : std::clamp(
                0.5 * kCrossoverRate +
                    0.5 * (0.45 + static_cast<double>(
                        operator_stats[0].improvements
                    ) / std::max(1, operator_stats[0].attempts)),
                0.55,
                0.95
            );
        const bool used_crossover =
            RandomChance(crossover_rate) && population.size() > 1;
        if (used_crossover) {
            bool cross_machine = RandomChance(0.65);
            bool cross_operation = RandomChance(0.65);
            if (!cross_machine && !cross_operation) {
                cross_operation = true;
            }
            if (cross_machine) {
                MachineCross(
                    first_parent.machine_code,
                    second_parent.machine_code,
                    first_child.machine_code,
                    second_child.machine_code
                );
            }
            if (cross_operation) {
                OperationCross(
                    first_parent.operation_code,
                    second_parent.operation_code,
                    first_child.operation_code,
                    second_child.operation_code
                );
            }
        }

        const bool first_mutated = MutateIndividual(
            first_child,
            jobs,
            jobList,
            mutation_rate,
            base_schedule,
            machines
        );
        const bool second_mutated = MutateIndividual(
            second_child,
            jobs,
            jobList,
            mutation_rate,
            base_schedule,
            machines
        );
        EvaluateIndividual(
            first_child,
            jobs,
            jobList,
            base_schedule,
            machines,
            fitness_cache
        );
        if (used_crossover) {
            const int parent_baseline = std::min(
                first_parent.fitness,
                second_parent.fitness
            );
            ++operator_stats[0].attempts;
            if (first_child.fitness < parent_baseline) {
                ++operator_stats[0].improvements;
                operator_stats[0].total_gain += static_cast<double>(
                    parent_baseline - first_child.fitness
                ) / std::max(1, parent_baseline);
            }
        }
        if (first_mutated) {
            ++operator_stats[1].attempts;
            const int parent_baseline = std::min(
                first_parent.fitness,
                second_parent.fitness
            );
            if (first_child.fitness < parent_baseline) {
                ++operator_stats[1].improvements;
                operator_stats[1].total_gain += static_cast<double>(
                    parent_baseline - first_child.fitness
                ) / std::max(1, parent_baseline);
            }
        }
        offspring.push_back(std::move(first_child));

        if (static_cast<int>(offspring.size()) < offspring_count &&
            !SearchDeadlineReached(deadline)) {
            EvaluateIndividual(
                second_child,
                jobs,
                jobList,
                base_schedule,
                machines,
                fitness_cache
            );
            if (used_crossover) {
                const int parent_baseline = std::min(
                    first_parent.fitness,
                    second_parent.fitness
                );
                ++operator_stats[0].attempts;
                if (second_child.fitness < parent_baseline) {
                    ++operator_stats[0].improvements;
                    operator_stats[0].total_gain += static_cast<double>(
                        parent_baseline - second_child.fitness
                    ) / std::max(1, parent_baseline);
                }
            }
            if (second_mutated) {
                ++operator_stats[1].attempts;
                const int parent_baseline = std::min(
                    first_parent.fitness,
                    second_parent.fitness
                );
                if (second_child.fitness < parent_baseline) {
                    ++operator_stats[1].improvements;
                    operator_stats[1].total_gain += static_cast<double>(
                        parent_baseline - second_child.fitness
                    ) / std::max(1, parent_baseline);
                }
            }
            offspring.push_back(std::move(second_child));
        }
    }
    return offspring;
}

std::optional<Individual> ImproveEliteWithGreedy(
    const Individual &individual,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines,
    std::unordered_map<std::string, int> &fitness_cache,
    const SearchDeadline deadline
) {
    if (SearchDeadlineReached(deadline)) {
        return std::nullopt;
    }
    Schedule decoded = DecodeSchedule(
        individual.machine_code,
        individual.operation_code,
        jobs,
        jobList,
        base_schedule,
        machines
    );
    Schedule improved = GreedySearch(
        decoded,
        jobs,
        jobList,
        1,
        machines,
        deadline
    );
    if (improved.get_TotalTime() >= individual.fitness) {
        return std::nullopt;
    }

    Individual improved_individual;
    EncodeSchedule(
        improved,
        jobs,
        jobList,
        improved_individual.machine_code,
        improved_individual.operation_code
    );
    EvaluateIndividual(
        improved_individual,
        jobs,
        jobList,
        base_schedule,
        machines,
        fitness_cache
    );
    if (improved_individual.fitness < individual.fitness) {
        return improved_individual;
    }
    return std::nullopt;
}

std::optional<Individual> ImproveEliteWithTabu(
    const Individual &individual,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines,
    std::unordered_map<std::string, int> &fitness_cache,
    const SearchDeadline deadline
) {
    if (SearchDeadlineReached(deadline)) {
        return std::nullopt;
    }
    Schedule decoded;
    try {
        decoded = DecodeSchedule(
            individual.machine_code,
            individual.operation_code,
            jobs,
            jobList,
            base_schedule,
            machines
        );
    } catch (const std::exception &) {
        return std::nullopt;
    }
    Schedule improved = TabuSearch(
        decoded,
        jobs,
        jobList,
        25,
        6,
        machines,
        deadline,
        false
    );
    if (improved.get_TotalTime() >= individual.fitness) {
        return std::nullopt;
    }

    Individual improved_individual;
    EncodeSchedule(
        improved,
        jobs,
        jobList,
        improved_individual.machine_code,
        improved_individual.operation_code
    );
    EvaluateIndividual(
        improved_individual,
        jobs,
        jobList,
        base_schedule,
        machines,
        fitness_cache
    );
    return improved_individual.fitness < individual.fitness
        ? std::optional<Individual>(std::move(improved_individual))
        : std::nullopt;
}

int GetImmigrantCount(const int population_size, const int stagnation_count) {
    if (stagnation_count >= kHighMutationStagnation) {
        return std::max(1, population_size / 3);
    }
    if (stagnation_count >= kElevatedMutationStagnation) {
        return std::max(1, population_size / 5);
    }
    return 0;
}

double GetMutationRate(const int stagnation_count) {
    if (stagnation_count >= kHighMutationStagnation) {
        return kHighMutationRate;
    }
    if (stagnation_count >= kElevatedMutationStagnation) {
        return kElevatedMutationRate;
    }
    return kBaseMutationRate;
}

double GetAdaptiveMutationRate(
    const int stagnation_count,
    const GeneticOperatorStats &stats
) {
    const double base_rate = GetMutationRate(stagnation_count);
    if (stats.attempts < 8) {
        return base_rate;
    }
    const double success_rate = static_cast<double>(stats.improvements) /
                                std::max(1, stats.attempts);
    return std::clamp(
        base_rate * (1.20 - 0.45 * success_rate),
        0.05,
        0.60
    );
}

void AddRandomImmigrants(
    std::vector<Individual> &population,
    const int target_size,
    const std::vector<int> &base_operation_code,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines,
    std::unordered_map<std::string, int> &fitness_cache,
    std::unordered_set<std::string> &population_keys,
    const SearchDeadline deadline
) {
    std::vector<std::vector<int>> machine_codes;
    std::vector<std::vector<int>> operation_codes;
    int attempts = 0;
    while (static_cast<int>(population.size()) < target_size &&
           attempts < target_size * 10 &&
           !SearchDeadlineReached(deadline)) {
        RandomInitializePopulation(
            jobs,
            machines,
            jobList,
            1,
            base_operation_code,
            machine_codes,
            operation_codes
        );
        if (!machine_codes.empty() && !operation_codes.empty()) {
            AddUniqueIndividual(
                {machine_codes.front(), operation_codes.front(), INT_MAX},
                population,
                population_keys,
                jobs,
                jobList,
                base_schedule,
                machines,
                fitness_cache
            );
        }
        ++attempts;
    }
}

std::vector<Individual> SelectNextGeneration(
    std::vector<Individual> candidates,
    const int population_size,
    const int immigrant_count,
    const std::vector<int> &base_operation_code,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines,
    std::unordered_map<std::string, int> &fitness_cache,
    const SearchDeadline deadline
) {
    std::stable_sort(
        candidates.begin(),
        candidates.end(),
        [](const Individual &left, const Individual &right) {
            return left.fitness < right.fitness;
        }
    );

    const int survivor_count = std::max(1, population_size - immigrant_count);
    std::vector<Individual> next_population;
    next_population.reserve(population_size);
    std::unordered_set<std::string> population_keys;
    population_keys.reserve(population_size * 2);
    const int unconditional_elites = std::min(2, survivor_count);
    for (const auto &candidate : candidates) {
        const std::string key = BuildChromosomeKey(candidate);
        if (population_keys.contains(key)) {
            continue;
        }
        if (static_cast<int>(next_population.size()) >= unconditional_elites &&
            !IsStructurallyDiverse(
                candidate,
                next_population,
                kMinimumPopulationDistance
            )) {
            continue;
        }
        population_keys.insert(key);
        next_population.push_back(candidate);
        if (static_cast<int>(next_population.size()) >= survivor_count) {
            break;
        }
    }

    AddRandomImmigrants(
        next_population,
        population_size,
        base_operation_code,
        jobs,
        jobList,
        base_schedule,
        machines,
        fitness_cache,
        population_keys,
        deadline
    );

    for (const auto &candidate : candidates) {
        if (static_cast<int>(next_population.size()) >= population_size) {
            break;
        }
        const std::string key = BuildChromosomeKey(candidate);
        if (population_keys.insert(key).second) {
            next_population.push_back(candidate);
        }
    }
    return next_population;
}

void UpdateBestSchedule(
    const Individual &individual,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &base_schedule,
    std::vector<Machine> &machines,
    int &best_fitness,
    Schedule &best_schedule
) {
    if (individual.fitness >= best_fitness) {
        return;
    }
    Schedule candidate = DecodeSchedule(
        individual.machine_code,
        individual.operation_code,
        jobs,
        jobList,
        base_schedule,
        machines
    );
    if (candidate.get_TotalTime() < best_fitness) {
        best_fitness = candidate.get_TotalTime();
        best_schedule = std::move(candidate);
    }
}

}  // namespace

void SetRandomSearchSeed(const unsigned int random_seed) {
    GetSharedRng().seed(random_seed);
}

Schedule RandomSearch(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    const std::vector<std::string> &jobList,
    const double strategy_param,
    const int repeat_count,
    const int max_repeat_count,
    const int max_iter_count,
    const int tabu_list_length,
    const int population_size,
    const SearchDeadline deadline
) {
    if (repeat_count < max_repeat_count / 2 || RandomChance(strategy_param)) {
        const int start_time = time(nullptr);
        Schedule result = AggressiveSearch(
            schedule,
            jobs,
            machines,
            jobList,
            population_size,
            max_iter_count,
            deadline
        );
        std::cout << "Genetic Algorithm time: "
                  << time(nullptr) - start_time << "s" << std::endl;
        return result;
    }

    const int start_time = time(nullptr);
    Schedule result = ConservativeSearch(
        schedule,
        jobs,
        jobList,
        tabu_list_length,
        max_iter_count,
        machines,
        deadline
    );
    std::cout << "Tabu Search time: "
              << time(nullptr) - start_time << "s" << std::endl;
    return result;
}

Schedule ConservativeSearch(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const int tabu_list_length,
    const int max_iter_count,
    std::vector<Machine> &machines,
    const SearchDeadline deadline
) {
    return TabuSearch(
        schedule,
        jobs,
        jobList,
        tabu_list_length,
        max_iter_count,
        machines,
        deadline
    );
}

Schedule AggressiveSearch(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    const std::vector<std::string> &jobList,
    int population_size,
    const int max_iter_count,
    const SearchDeadline deadline
) {
    if (jobs.empty() || jobList.empty() || max_iter_count <= 0 ||
        SearchDeadlineReached(deadline)) {
        return schedule;
    }
    population_size = std::max(4, population_size);

    PersistentGeneticState &state = GetPersistentGeneticState();
    const std::string problem_signature = BuildProblemSignature(jobs, jobList);
    if (state.problem_signature != problem_signature || state.population.empty()) {
        state = PersistentGeneticState{};
        state.problem_signature = problem_signature;
        state.fitness_cache.reserve(
            std::max(32, population_size * std::max(1, max_iter_count))
        );
        state.population = BuildInitialPopulation(
            schedule,
            jobs,
            machines,
            jobList,
            population_size,
            state.fitness_cache,
            deadline
        );
    }

    std::unordered_map<std::string, int> &fitness_cache = state.fitness_cache;
    for (auto &operator_stats : state.operator_stats) {
        if (operator_stats.attempts >= 200) {
            operator_stats.attempts /= 2;
            operator_stats.improvements /= 2;
            operator_stats.total_gain *= 0.5;
        }
    }
    std::vector<Individual> population = std::move(state.population);
    const int previous_population_best = population.empty()
        ? INT_MAX
        : std::min_element(
              population.begin(),
              population.end(),
              [](const Individual &left, const Individual &right) {
                  return left.fitness < right.fitness;
              }
          )->fitness;
    std::unordered_set<std::string> population_keys;
    population_keys.reserve(population_size * 2);
    for (const auto &individual : population) {
        population_keys.insert(BuildChromosomeKey(individual));
    }

    std::vector<int> incumbent_machine_code;
    std::vector<int> incumbent_operation_code;
    EncodeSchedule(
        schedule,
        jobs,
        jobList,
        incumbent_machine_code,
        incumbent_operation_code
    );
    AddUniqueIndividual(
        {incumbent_machine_code, incumbent_operation_code, INT_MAX},
        population,
        population_keys,
        jobs,
        jobList,
        schedule,
        machines,
        fitness_cache
    );
    if (schedule.get_TotalTime() < previous_population_best) {
        state.stagnation_count = 0;
    }
    std::stable_sort(
        population.begin(),
        population.end(),
        [](const Individual &left, const Individual &right) {
            return left.fitness < right.fitness;
        }
    );
    if (static_cast<int>(population.size()) > population_size) {
        population.resize(population_size);
        population_keys.clear();
        for (const auto &individual : population) {
            population_keys.insert(BuildChromosomeKey(individual));
        }
    }
    if (!population.empty() &&
        (state.calls_without_population_improvement >= 4 ||
         PopulationDiversity(population) < kPopulationRestartDiversity)) {
        const int elite_count = std::min(
            static_cast<int>(population.size()),
            std::max(2, population_size / 5)
        );
        population.resize(elite_count);
        population_keys.clear();
        for (const auto &individual : population) {
            population_keys.insert(BuildChromosomeKey(individual));
        }
        AddRandomImmigrants(
            population,
            population_size,
            incumbent_operation_code,
            jobs,
            jobList,
            schedule,
            machines,
            fitness_cache,
            population_keys,
            deadline
        );
        state.operator_stats = {};
        state.stagnation_count = kElevatedMutationStagnation;
        state.calls_without_population_improvement = 0;
    }
    while (static_cast<int>(population.size()) < population_size &&
           !population.empty() && !SearchDeadlineReached(deadline)) {
        AddRandomImmigrants(
            population,
            population_size,
            incumbent_operation_code,
            jobs,
            jobList,
            schedule,
            machines,
            fitness_cache,
            population_keys,
            deadline
        );
    }
    if (population.empty()) {
        state.population.clear();
        return schedule;
    }

    int best_fitness = schedule.get_TotalTime();
    Schedule best_schedule = schedule;
    for (const auto &individual : population) {
        if (SearchDeadlineReached(deadline)) {
            break;
        }
        UpdateBestSchedule(
            individual,
            jobs,
            jobList,
            schedule,
            machines,
            best_fitness,
            best_schedule
        );
    }

    std::vector<int> base_machine_code;
    std::vector<int> base_operation_code;
    EncodeSchedule(
        schedule,
        jobs,
        jobList,
        base_machine_code,
        base_operation_code
    );

    int stagnation_count = state.stagnation_count;
    int generations_completed = 0;
    for (int generation = 0;
         generation < max_iter_count && !SearchDeadlineReached(deadline);
         ++generation) {
        const int absolute_generation = state.generation + generation;
        const int best_before_generation = best_fitness;
        const double mutation_rate = GetAdaptiveMutationRate(
            stagnation_count,
            state.operator_stats[1]
        );
        std::vector<Individual> offspring = BuildOffspring(
            population,
            population_size,
            mutation_rate,
            jobs,
            jobList,
            schedule,
            machines,
            fitness_cache,
            deadline,
            state.operator_stats
        );

        if (!SearchDeadlineReached(deadline) && !offspring.empty() &&
            ((absolute_generation + 1) % kLocalSearchInterval == 0 ||
             generation + 1 == max_iter_count)) {
            std::vector<int> elite_indices(offspring.size());
            std::iota(elite_indices.begin(), elite_indices.end(), 0);
            std::stable_sort(
                elite_indices.begin(),
                elite_indices.end(),
                [&offspring](const int left, const int right) {
                    return offspring[left].fitness < offspring[right].fitness;
                }
            );
            const int local_search_count = generation + 1 == max_iter_count
                ? std::min(2, static_cast<int>(elite_indices.size()))
                : 1;
            for (int elite = 0;
                 elite < local_search_count &&
                     !SearchDeadlineReached(deadline);
                 ++elite) {
                const Individual &elite_individual =
                    offspring[elite_indices[elite]];
                if (generation + 1 != max_iter_count &&
                    elite_individual.fitness > best_fitness) {
                    continue;
                }
                const auto improved_elite = elite % 2 == 0
                    ? ImproveEliteWithGreedy(
                        elite_individual,
                        jobs,
                        jobList,
                        schedule,
                        machines,
                        fitness_cache,
                        deadline
                    )
                    : ImproveEliteWithTabu(
                        elite_individual,
                        jobs,
                        jobList,
                        schedule,
                        machines,
                        fitness_cache,
                        deadline
                    );
                if (improved_elite.has_value()) {
                    offspring.push_back(*improved_elite);
                }
            }
        }

        for (const auto &individual : offspring) {
            UpdateBestSchedule(
                individual,
                jobs,
                jobList,
                schedule,
                machines,
                best_fitness,
                best_schedule
            );
        }

        if (SearchDeadlineReached(deadline)) {
            break;
        }

        std::vector<Individual> candidates = population;
        candidates.insert(candidates.end(), offspring.begin(), offspring.end());
        const int immigrant_count = GetImmigrantCount(population_size, stagnation_count);
        population = SelectNextGeneration(
            std::move(candidates),
            population_size,
            immigrant_count,
            base_operation_code,
            jobs,
            jobList,
            schedule,
            machines,
            fitness_cache,
            deadline
        );

        for (const auto &individual : population) {
            UpdateBestSchedule(
                individual,
                jobs,
                jobList,
                schedule,
                machines,
                best_fitness,
                best_schedule
            );
        }

        if (best_fitness < best_before_generation) {
            stagnation_count = 0;
        } else {
            ++stagnation_count;
        }
        ++generations_completed;
    }
    state.population = std::move(population);
    state.generation += generations_completed;
    state.stagnation_count = stagnation_count;
    const int improvement_reference = std::min(
        schedule.get_TotalTime(),
        previous_population_best
    );
    if (best_fitness < improvement_reference) {
        state.calls_without_population_improvement = 0;
    } else {
        ++state.calls_without_population_improvement;
    }
    return best_schedule;
}

void EncodeSchedule(
    const Schedule &schedule,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    std::vector<int> &machine_selection_code,
    std::vector<int> &operation_sequencing_code
) {
    machine_selection_code.clear();
    operation_sequencing_code.clear();
    const auto job_name_map = BuildJobNameMap(jobs);
    const auto ordered_jobs = BuildOrderedJobsByJobList(jobList, job_name_map);
    const auto schedule_machine_map = BuildScheduleMachineMap(schedule);

    for (const Job *job : ordered_jobs) {
        if (job == nullptr) {
            continue;
        }
        for (const auto &job_process : job->get_job_process()) {
            if (job_process.process_item.empty()) {
                continue;
            }
            const int process_id = job_process.process_item.front().process_id;
            const auto machine_it = schedule_machine_map.find(
                BuildProcessId(job->get_job_id(), process_id)
            );
            const int selected_machine = machine_it == schedule_machine_map.end()
                ? job_process.process_item.front().machine_id
                : machine_it->second;
            int selected_option = 0;
            for (int option = 0; option < job_process.machine_count; ++option) {
                if (job_process.process_item[option].machine_id == selected_machine) {
                    selected_option = option;
                    break;
                }
            }
            machine_selection_code.push_back(selected_option);
        }
    }

    operation_sequencing_code = BuildTopologicalOperationCode(schedule);
    if (operation_sequencing_code.size() != machine_selection_code.size()) {
        operation_sequencing_code.clear();
        for (const Job *job : ordered_jobs) {
            if (job == nullptr) {
                continue;
            }
            for (int process = 0; process < job->get_process_count(); ++process) {
                operation_sequencing_code.push_back(job->get_job_id());
            }
        }
    }
}

Schedule DecodeSchedule(
    const std::vector<int> &machine_selection_code,
    const std::vector<int> &operation_sequencing_code,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &schedule,
    std::vector<Machine> &machines
) {
    const auto job_name_map = BuildJobNameMap(jobs);
    const auto job_id_map = BuildJobIdMap(jobs);
    const auto ordered_jobs = BuildOrderedJobsByJobList(jobList, job_name_map);
    int maximum_job_id = -1;
    for (const auto &job : jobs) {
        maximum_job_id = std::max(maximum_job_id, job.get_job_id());
    }

    std::vector<std::vector<int>> machine_matrix(maximum_job_id + 1);
    std::vector<int> now_process_index(maximum_job_id + 1, 0);
    for (const auto &job : jobs) {
        machine_matrix[job.get_job_id()].assign(job.get_process_count(), -1);
    }

    size_t machine_code_index = 0;
    for (const Job *job : ordered_jobs) {
        if (job == nullptr) {
            continue;
        }
        for (const auto &job_process : job->get_job_process()) {
            if (machine_code_index >= machine_selection_code.size() ||
                job_process.process_item.empty()) {
                throw std::runtime_error("Invalid machine-selection chromosome length.");
            }
            const int selected_option = machine_selection_code[machine_code_index++];
            if (selected_option < 0 || selected_option >= job_process.machine_count) {
                throw std::runtime_error("Invalid machine option in chromosome.");
            }
            const int process_id = job_process.process_item.front().process_id;
            machine_matrix[job->get_job_id()][process_id] =
                job_process.process_item[selected_option].machine_id;
        }
    }

    Schedule result;
    result.set_schedule_id(schedule.get_schedule_id() + 1);
    result.set_machine_count(schedule.get_machine_count());
    std::vector<Schedule_item> schedule_items(schedule.get_machine_count());
    for (int machine_id = 0; machine_id < schedule.get_machine_count(); ++machine_id) {
        schedule_items[machine_id].machine_id = machine_id;
        schedule_items[machine_id].process_count = 0;
    }

    for (const int job_id : operation_sequencing_code) {
        const auto job_it = job_id_map.find(job_id);
        if (job_it == job_id_map.end() || job_id < 0 || job_id > maximum_job_id) {
            throw std::runtime_error("Invalid job id in operation-sequencing chromosome.");
        }
        const int process_index = now_process_index[job_id]++;
        if (process_index >= job_it->second->get_process_count()) {
            throw std::runtime_error("Too many job occurrences in operation-sequencing chromosome.");
        }
        const int machine_id = machine_matrix[job_id][process_index];
        if (machine_id < 0 || machine_id >= schedule.get_machine_count()) {
            throw std::runtime_error("Invalid decoded machine id.");
        }
        schedule_items[machine_id].schedule_process.push_back({job_id, process_index});
        ++schedule_items[machine_id].process_count;
    }

    for (const auto &[job_id, job] : job_id_map) {
        if (now_process_index[job_id] != job->get_process_count()) {
            throw std::runtime_error("Missing job occurrence in operation-sequencing chromosome.");
        }
    }

    result.set_schedule_items(schedule_items);
    ScheduleItemsToGraph(result, schedule_items, jobs, jobList, true, machines);
    return result;
}

int GetMachineIdByJobIdAndProcessId(
    const Schedule &schedule,
    const int job_id,
    const int process_id
) {
    for (const auto &schedule_item : schedule.get_schedule_items()) {
        for (const auto &process : schedule_item.schedule_process) {
            if (process.job_id == job_id && process.process_id == process_id) {
                return schedule_item.machine_id;
            }
        }
    }
    return -1;
}

void GlobalInitializePopulation(
    const std::vector<Job> &jobs,
    const std::vector<Machine> &machines,
    const std::vector<std::string> &jobList,
    const int population_size,
    const std::vector<int> &operation_sequencing_code,
    std::vector<std::vector<int>> &population_machine_selection_code,
    std::vector<std::vector<int>> &population_operation_sequencing_code
) {
    population_machine_selection_code.clear();
    population_operation_sequencing_code.clear();
    if (population_size <= 0) {
        return;
    }

    const auto job_name_map = BuildJobNameMap(jobs);
    const auto ordered_jobs = BuildOrderedJobsByJobList(jobList, job_name_map);
    std::vector<int> job_offsets(ordered_jobs.size(), 0);
    int operation_count = 0;
    for (int index = 0; index < static_cast<int>(ordered_jobs.size()); ++index) {
        job_offsets[index] = operation_count;
        if (ordered_jobs[index] != nullptr) {
            operation_count += ordered_jobs[index]->get_process_count();
        }
    }

    population_machine_selection_code.reserve(population_size);
    population_operation_sequencing_code.reserve(population_size);
    for (int individual = 0; individual < population_size; ++individual) {
        std::vector<int> machine_code(operation_count, 0);
        std::vector<int> job_indices(ordered_jobs.size());
        std::iota(job_indices.begin(), job_indices.end(), 0);
        std::shuffle(job_indices.begin(), job_indices.end(), GetSharedRng());
        std::vector<int> machine_completion(machines.size(), 0);

        for (const int job_index : job_indices) {
            const Job *job = ordered_jobs[job_index];
            if (job == nullptr) {
                continue;
            }
            int process_offset = 0;
            for (const auto &job_process : job->get_job_process()) {
                int best_option = 0;
                int best_completion = INT_MAX;
                for (int option = 0; option < job_process.machine_count; ++option) {
                    const auto &process_item = job_process.process_item[option];
                    const int completion = machine_completion[process_item.machine_id] +
                                           process_item.process_time;
                    if (completion < best_completion ||
                        (completion == best_completion && RandomChance(0.5))) {
                        best_completion = completion;
                        best_option = option;
                    }
                }
                machine_code[job_offsets[job_index] + process_offset] = best_option;
                const auto &selected_item = job_process.process_item[best_option];
                machine_completion[selected_item.machine_id] += selected_item.process_time;
                ++process_offset;
            }
        }

        std::vector<int> operation_code = operation_sequencing_code;
        std::shuffle(operation_code.begin(), operation_code.end(), GetSharedRng());
        population_machine_selection_code.push_back(std::move(machine_code));
        population_operation_sequencing_code.push_back(std::move(operation_code));
    }
}

void LocalInitializePopulation(
    const std::vector<Job> &jobs,
    const std::vector<Machine> &machines,
    const std::vector<std::string> &jobList,
    const int population_size,
    const std::vector<int> &operation_sequencing_code,
    std::vector<std::vector<int>> &population_machine_selection_code,
    std::vector<std::vector<int>> &population_operation_sequencing_code
) {
    population_machine_selection_code.clear();
    population_operation_sequencing_code.clear();
    if (population_size <= 0) {
        return;
    }

    const auto job_name_map = BuildJobNameMap(jobs);
    const auto ordered_jobs = BuildOrderedJobsByJobList(jobList, job_name_map);
    population_machine_selection_code.reserve(population_size);
    population_operation_sequencing_code.reserve(population_size);
    for (int individual = 0; individual < population_size; ++individual) {
        std::vector<int> machine_code;
        machine_code.reserve(operation_sequencing_code.size());
        for (const Job *job : ordered_jobs) {
            if (job == nullptr) {
                continue;
            }
            std::vector<int> local_machine_completion(machines.size(), 0);
            for (const auto &job_process : job->get_job_process()) {
                int best_option = 0;
                int best_completion = INT_MAX;
                for (int option = 0; option < job_process.machine_count; ++option) {
                    const auto &process_item = job_process.process_item[option];
                    const int completion = local_machine_completion[process_item.machine_id] +
                                           process_item.process_time;
                    if (completion < best_completion ||
                        (completion == best_completion && RandomChance(0.5))) {
                        best_completion = completion;
                        best_option = option;
                    }
                }
                machine_code.push_back(best_option);
                const auto &selected_item = job_process.process_item[best_option];
                local_machine_completion[selected_item.machine_id] += selected_item.process_time;
            }
        }

        std::vector<int> operation_code = operation_sequencing_code;
        std::shuffle(operation_code.begin(), operation_code.end(), GetSharedRng());
        population_machine_selection_code.push_back(std::move(machine_code));
        population_operation_sequencing_code.push_back(std::move(operation_code));
    }
}

void RandomInitializePopulation(
    const std::vector<Job> &jobs,
    const std::vector<Machine> &machines,
    const std::vector<std::string> &jobList,
    const int population_size,
    const std::vector<int> &operation_sequencing_code,
    std::vector<std::vector<int>> &population_machine_selection_code,
    std::vector<std::vector<int>> &population_operation_sequencing_code
) {
    (void)machines;
    population_machine_selection_code.clear();
    population_operation_sequencing_code.clear();
    if (population_size <= 0) {
        return;
    }

    const auto job_name_map = BuildJobNameMap(jobs);
    const auto ordered_jobs = BuildOrderedJobsByJobList(jobList, job_name_map);
    population_machine_selection_code.reserve(population_size);
    population_operation_sequencing_code.reserve(population_size);
    for (int individual = 0; individual < population_size; ++individual) {
        std::vector<int> machine_code;
        machine_code.reserve(operation_sequencing_code.size());
        for (const Job *job : ordered_jobs) {
            if (job == nullptr) {
                continue;
            }
            for (const auto &job_process : job->get_job_process()) {
                machine_code.push_back(RandomIndex(job_process.machine_count));
            }
        }
        std::vector<int> operation_code = operation_sequencing_code;
        std::shuffle(operation_code.begin(), operation_code.end(), GetSharedRng());
        population_machine_selection_code.push_back(std::move(machine_code));
        population_operation_sequencing_code.push_back(std::move(operation_code));
    }
}

void MachineCross(
    const std::vector<int> &parent1,
    const std::vector<int> &parent2,
    std::vector<int> &child1,
    std::vector<int> &child2
) {
    child1 = parent1;
    child2 = parent2;
    const int gene_count = std::min(parent1.size(), parent2.size());
    if (gene_count <= 1) {
        return;
    }

    std::vector<int> indices(gene_count);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), GetSharedRng());
    const int exchange_count = std::uniform_int_distribution<int>(1, gene_count - 1)(GetSharedRng());
    for (int index = 0; index < exchange_count; ++index) {
        const int gene = indices[index];
        child1[gene] = parent2[gene];
        child2[gene] = parent1[gene];
    }
}

void OperationCross(
    const std::vector<int> &parent1,
    const std::vector<int> &parent2,
    std::vector<int> &child1,
    std::vector<int> &child2
) {
    child1 = parent1;
    child2 = parent2;
    if (parent1.size() != parent2.size() || parent1.size() <= 1) {
        return;
    }

    std::vector<int> job_ids = parent1;
    std::sort(job_ids.begin(), job_ids.end());
    job_ids.erase(std::unique(job_ids.begin(), job_ids.end()), job_ids.end());
    if (job_ids.size() <= 1) {
        return;
    }
    std::shuffle(job_ids.begin(), job_ids.end(), GetSharedRng());
    const int selected_count = std::uniform_int_distribution<int>(
        1,
        static_cast<int>(job_ids.size()) - 1
    )(GetSharedRng());
    std::unordered_set<int> selected_jobs(
        job_ids.begin(),
        job_ids.begin() + selected_count
    );

    child1.assign(parent1.size(), -1);
    child2.assign(parent2.size(), -1);
    for (int index = 0; index < static_cast<int>(parent1.size()); ++index) {
        if (selected_jobs.contains(parent1[index])) {
            child1[index] = parent1[index];
        }
        if (!selected_jobs.contains(parent2[index])) {
            child2[index] = parent2[index];
        }
    }

    int first_fill = 0;
    int second_fill = 0;
    for (const int job_id : parent2) {
        if (selected_jobs.contains(job_id)) {
            continue;
        }
        while (child1[first_fill] != -1) {
            ++first_fill;
        }
        child1[first_fill] = job_id;
    }
    for (const int job_id : parent1) {
        if (!selected_jobs.contains(job_id)) {
            continue;
        }
        while (child2[second_fill] != -1) {
            ++second_fill;
        }
        child2[second_fill] = job_id;
    }
}

void MachineVariation(
    const std::vector<int> &parent,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    std::vector<int> &child
) {
    child = parent;
    const auto ordered_processes = BuildOrderedProcesses(jobs, jobList);
    const int gene_count = std::min(child.size(), ordered_processes.size());
    std::vector<int> flexible_genes;
    flexible_genes.reserve(gene_count);
    for (int gene = 0; gene < gene_count; ++gene) {
        if (ordered_processes[gene] != nullptr &&
            ordered_processes[gene]->machine_count > 1) {
            flexible_genes.push_back(gene);
        }
    }
    if (flexible_genes.empty()) {
        return;
    }

    std::shuffle(flexible_genes.begin(), flexible_genes.end(), GetSharedRng());
    const int maximum_mutations = std::min(3, static_cast<int>(flexible_genes.size()));
    const int mutation_count = std::uniform_int_distribution<int>(1, maximum_mutations)(GetSharedRng());
    for (int mutation = 0; mutation < mutation_count; ++mutation) {
        const int gene = flexible_genes[mutation];
        const Job_process &job_process = *ordered_processes[gene];
        std::vector<int> alternatives;
        alternatives.reserve(job_process.machine_count - 1);
        for (int option = 0; option < job_process.machine_count; ++option) {
            if (option != child[gene]) {
                alternatives.push_back(option);
            }
        }
        std::sort(
            alternatives.begin(),
            alternatives.end(),
            [&job_process](const int left, const int right) {
                return job_process.process_item[left].process_time <
                       job_process.process_item[right].process_time;
            }
        );

        if (RandomChance(0.75)) {
            const int candidate_count = std::min(2, static_cast<int>(alternatives.size()));
            child[gene] = alternatives[RandomIndex(candidate_count)];
        } else {
            child[gene] = alternatives[RandomIndex(static_cast<int>(alternatives.size()))];
        }
    }
}

void OperationVariation(const std::vector<int> &parent, std::vector<int> &child) {
    child = parent;
    const int gene_count = static_cast<int>(child.size());
    if (gene_count <= 1) {
        return;
    }

    int first_position = RandomIndex(gene_count);
    int second_position = RandomIndex(gene_count - 1);
    if (second_position >= first_position) {
        ++second_position;
    }
    for (int attempt = 0;
         attempt < 8 && child[first_position] == child[second_position];
         ++attempt) {
        second_position = RandomIndex(gene_count - 1);
        if (second_position >= first_position) {
            ++second_position;
        }
    }
    if (child[first_position] == child[second_position]) {
        return;
    }

    const int mutation_type = RandomIndex(3);
    if (mutation_type == 0) {
        std::swap(child[first_position], child[second_position]);
    } else if (mutation_type == 1) {
        const int moved_job = child[first_position];
        child.erase(child.begin() + first_position);
        child.insert(child.begin() + second_position, moved_job);
    } else {
        int first = std::min(first_position, second_position);
        int second = std::max(first_position, second_position);
        const int maximum_span = std::max(2, gene_count / 10);
        second = std::min(second, first + maximum_span);
        if (first != second) {
            std::reverse(child.begin() + first, child.begin() + second + 1);
        }
    }
}

int CalculateFitness(
    const std::vector<int> &machine_selection_code,
    const std::vector<int> &operation_sequencing_code,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &jobList,
    const Schedule &schedule0,
    std::vector<Machine> &machines,
    std::unordered_map<std::string, int> &fitness_cache
) {
    const std::string cache_key = BuildChromosomeKey(
        machine_selection_code,
        operation_sequencing_code
    );
    const auto cache_it = fitness_cache.find(cache_key);
    if (cache_it != fitness_cache.end()) {
        return cache_it->second;
    }
    Schedule schedule = DecodeSchedule(
        machine_selection_code,
        operation_sequencing_code,
        jobs,
        jobList,
        schedule0,
        machines
    );
    const int fitness = schedule.get_TotalTime();
    fitness_cache.emplace(cache_key, fitness);
    return fitness;
}
