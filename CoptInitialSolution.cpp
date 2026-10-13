#include "CoptInitialSolution.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "DataProc.h"
#include "callbackbase.h"
#include "coptcpp_pch.h"

namespace {

class CoptIncumbentProgressCallback final : public CallbackBase {
public:
    explicit CoptIncumbentProgressCallback(const char *progress_path)
        : started_at_(std::chrono::steady_clock::now()),
          progress_stream_(progress_path, std::ios::out | std::ios::app) {}

    void callback() override {
        if (Where() != COPT_CBCONTEXT_INCUMBENT) {
            return;
        }

        const double incumbent = GetDblInfo(COPT_CBINFO_BESTOBJ);
        if (!std::isfinite(incumbent) || !progress_stream_.is_open()) {
            return;
        }

        const double elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_at_
        ).count();
        std::lock_guard<std::mutex> lock(stream_mutex_);
        progress_stream_ << std::fixed << std::setprecision(6)
                         << elapsed_seconds << ','
                         << std::llround(incumbent) << '\n'
                         << std::flush;
    }

private:
    std::chrono::steady_clock::time_point started_at_;
    std::ofstream progress_stream_;
    std::mutex stream_mutex_;
};

struct OperationInfo {
    int job_id;
    int process_id;
    std::vector<Process_item> options;
};

std::string MakeName(const char *prefix, int first, int second, int third = -1) {
    std::ostringstream name;
    name << prefix << "_" << first << "_" << second;
    if (third >= 0) {
        name << "_" << third;
    }
    return name.str();
}

std::vector<OperationInfo> BuildOperations(
    const std::vector<Job> &jobs,
    std::unordered_map<int, std::vector<int>> &job_operation_indices,
    double &horizon
) {
    std::vector<OperationInfo> operations;
    horizon = 0.0;

    for (const Job &job : jobs) {
        std::vector<int> &job_indices = job_operation_indices[job.get_job_id()];
        const auto &job_processes = job.get_job_process();
        job_indices.reserve(job_processes.size());

        for (const Job_process &job_process : job_processes) {
            if (job_process.process_item.empty()) {
                throw std::runtime_error(
                    "COPT initial solution: an operation has no eligible machine."
                );
            }

            double maximum_process_time = 0.0;
            for (const Process_item &option : job_process.process_item) {
                if (option.process_time < 0) {
                    throw std::runtime_error(
                        "COPT initial solution: processing time cannot be negative."
                    );
                }
                maximum_process_time = std::max(
                    maximum_process_time,
                    static_cast<double>(option.process_time)
                );
            }

            const int operation_index = static_cast<int>(operations.size());
            operations.push_back({
                job.get_job_id(),
                job_process.process_item.front().process_id,
                job_process.process_item
            });
            job_indices.push_back(operation_index);
            horizon += maximum_process_time;
        }
    }

    return operations;
}

std::vector<std::string> BuildEffectiveJobList(
    const std::vector<std::string> &jobList,
    const std::vector<Job> &jobs
) {
    if (!jobList.empty()) {
        return jobList;
    }

    std::vector<std::string> effective_job_list;
    effective_job_list.reserve(jobs.size());
    for (const Job &job : jobs) {
        effective_job_list.push_back(job.get_job_name());
    }
    return effective_job_list;
}

} // namespace

Schedule GenerateCoptInitialSolution(
    const std::vector<std::string> &jobList,
    const std::vector<Job> &jobs,
    const std::vector<Machine> &machines,
    const double copt_time_limit_seconds
) {
    if (jobs.empty() || machines.empty()) {
        throw std::runtime_error(
            "COPT initial solution: jobs and machines must not be empty."
        );
    }
    if (!(copt_time_limit_seconds > 0.0) || !std::isfinite(copt_time_limit_seconds)) {
        throw std::runtime_error(
            "COPT initial solution: time limit must be a finite positive number."
        );
    }

    std::unordered_map<int, std::vector<int>> job_operation_indices;
    double horizon = 0.0;
    const std::vector<OperationInfo> operations = BuildOperations(
        jobs,
        job_operation_indices,
        horizon
    );
    if (operations.empty()) {
        throw std::runtime_error("COPT initial solution: no operations were found.");
    }
    horizon = std::max(1.0, horizon);

    for (const OperationInfo &operation : operations) {
        for (const Process_item &option : operation.options) {
            if (option.machine_id < 0 || option.machine_id >= static_cast<int>(machines.size())) {
                throw std::runtime_error(
                    "COPT initial solution: machine ids must be zero-based and within the input machine count."
                );
            }
        }
    }

    std::vector<Machine> graph_machines = machines;
    const int operation_count = static_cast<int>(operations.size());
    const int machine_count = static_cast<int>(machines.size());

    Envr env;
    Model model = env.CreateModel("HLS_FJSP_COPT_Initial_Solution");
    model.SetIntParam(COPT_INTPARAM_LOGGING, 0);

    std::unique_ptr<CoptIncumbentProgressCallback> progress_callback;
    const char *progress_path = std::getenv("HLS_COPT_PROGRESS_FILE");
    if (progress_path != nullptr && *progress_path != '\0') {
        progress_callback = std::make_unique<CoptIncumbentProgressCallback>(
            progress_path
        );
        model.SetCallback(progress_callback.get(), COPT_CBCONTEXT_INCUMBENT);
    }

    std::vector<Var> start_vars;
    std::vector<Var> finish_vars;
    std::vector<std::vector<Var>> assignment_vars;
    start_vars.reserve(operations.size());
    finish_vars.reserve(operations.size());
    assignment_vars.reserve(operations.size());

    for (int operation_index = 0; operation_index < operation_count; ++operation_index) {
        const OperationInfo &operation = operations[operation_index];
        start_vars.push_back(model.AddVar(
            0.0,
            horizon,
            0.0,
            COPT_CONTINUOUS,
            MakeName("start", operation.job_id, operation.process_id).c_str()
        ));
        finish_vars.push_back(model.AddVar(
            0.0,
            horizon,
            0.0,
            COPT_CONTINUOUS,
            MakeName("finish", operation.job_id, operation.process_id).c_str()
        ));

        std::vector<Var> operation_assignments;
        operation_assignments.reserve(operation.options.size());
        Expr assignment_sum;
        Expr processing_time_expression(start_vars.back());
        for (int option_index = 0; option_index < static_cast<int>(operation.options.size()); ++option_index) {
            const Process_item &option = operation.options[option_index];
            const std::string name = MakeName(
                "assign",
                operation.job_id,
                operation.process_id,
                option_index
            );
            Var assignment = model.AddVar(
                0.0,
                1.0,
                0.0,
                COPT_BINARY,
                name.c_str()
            );
            operation_assignments.push_back(assignment);
            assignment_sum += assignment;
            processing_time_expression += option.process_time * assignment;
        }
        model.AddConstr(
            assignment_sum == 1.0,
            MakeName("select", operation.job_id, operation.process_id).c_str()
        );
        model.AddConstr(
            finish_vars.back() == processing_time_expression,
            MakeName("duration", operation.job_id, operation.process_id).c_str()
        );
        assignment_vars.push_back(std::move(operation_assignments));
    }

    for (const auto &[job_id, operation_indices] : job_operation_indices) {
        for (int i = 1; i < static_cast<int>(operation_indices.size()); ++i) {
            const int previous = operation_indices[i - 1];
            const int current = operation_indices[i];
            model.AddConstr(
                start_vars[current] >= finish_vars[previous],
                MakeName("precedence", job_id, i).c_str()
            );
        }
    }

    Var makespan = model.AddVar(
        0.0,
        horizon,
        0.0,
        COPT_CONTINUOUS,
        "makespan"
    );
    for (int operation_index = 0; operation_index < operation_count; ++operation_index) {
        model.AddConstr(
            makespan >= finish_vars[operation_index],
            MakeName("makespan_bound", operations[operation_index].job_id, operations[operation_index].process_id).c_str()
        );
    }

    // For each pair of operations that can use the same machine, order them
    // with one binary variable. The assignment terms deactivate the pair
    // constraints when either operation is assigned to another machine.
    for (int first = 0; first < operation_count; ++first) {
        for (int second = first + 1; second < operation_count; ++second) {
            const OperationInfo &first_operation = operations[first];
            const OperationInfo &second_operation = operations[second];
            for (int first_option = 0; first_option < static_cast<int>(first_operation.options.size()); ++first_option) {
                const int machine_id = first_operation.options[first_option].machine_id;
                for (int second_option = 0; second_option < static_cast<int>(second_operation.options.size()); ++second_option) {
                    if (second_operation.options[second_option].machine_id != machine_id) {
                        continue;
                    }

                    const std::string order_name = MakeName(
                        "order",
                        first,
                        second,
                        machine_id
                    );
                    Var first_before_second = model.AddVar(
                        0.0,
                        1.0,
                        0.0,
                        COPT_BINARY,
                        order_name.c_str()
                    );

                    Expr first_order = start_vars[second] - finish_vars[first];
                    first_order += 3.0 * horizon;
                    first_order -= horizon * first_before_second;
                    first_order -= horizon * assignment_vars[first][first_option];
                    first_order -= horizon * assignment_vars[second][second_option];
                    model.AddConstr(
                        first_order >= 0.0,
                        MakeName("order_forward", first, second, machine_id).c_str()
                    );

                    Expr second_order = start_vars[first] - finish_vars[second];
                    second_order += 2.0 * horizon;
                    second_order += horizon * first_before_second;
                    second_order -= horizon * assignment_vars[first][first_option];
                    second_order -= horizon * assignment_vars[second][second_option];
                    model.AddConstr(
                        second_order >= 0.0,
                        MakeName("order_backward", first, second, machine_id).c_str()
                    );
                }
            }
        }
    }

    model.SetObjective(makespan, COPT_MINIMIZE);
    model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, copt_time_limit_seconds);
    model.Solve();

    if (model.GetIntAttr(COPT_INTATTR_HASSOL) == 0) {
        const int status = model.GetIntAttr(COPT_INTATTR_STATUS);
        std::ostringstream message;
        message << "COPT did not produce a feasible initial solution (status="
                << status << ").";
        throw std::runtime_error(message.str());
    }

    Schedule schedule;
    schedule.set_schedule_id(0);
    schedule.set_machine_count(machine_count);
    std::vector<Schedule_item> schedule_items(machine_count);
    for (int machine_id = 0; machine_id < machine_count; ++machine_id) {
        schedule_items[machine_id].machine_id = machine_id;
        schedule_items[machine_id].process_count = 0;
    }

    struct ScheduledOperation {
        int operation_index;
        double start_time;
    };
    std::vector<std::vector<ScheduledOperation>> machine_operations(machine_count);
    for (int operation_index = 0; operation_index < operation_count; ++operation_index) {
        const OperationInfo &operation = operations[operation_index];
        int selected_option = -1;
        for (int option_index = 0; option_index < static_cast<int>(assignment_vars[operation_index].size()); ++option_index) {
            if (assignment_vars[operation_index][option_index].Get(COPT_DBLINFO_VALUE) > 0.5) {
                selected_option = option_index;
                break;
            }
        }
        if (selected_option < 0) {
            throw std::runtime_error(
                "COPT returned a solution without a machine assignment."
            );
        }

        const int machine_id = operation.options[selected_option].machine_id;
        if (machine_id < 0 || machine_id >= machine_count) {
            throw std::runtime_error(
                "COPT returned an invalid machine assignment."
            );
        }
        machine_operations[machine_id].push_back({
            operation_index,
            start_vars[operation_index].Get(COPT_DBLINFO_VALUE)
        });
    }

    for (int machine_id = 0; machine_id < machine_count; ++machine_id) {
        auto &operations_on_machine = machine_operations[machine_id];
        std::sort(
            operations_on_machine.begin(),
            operations_on_machine.end(),
            [&operations](const ScheduledOperation &left, const ScheduledOperation &right) {
                if (std::fabs(left.start_time - right.start_time) > 1e-7) {
                    return left.start_time < right.start_time;
                }
                const OperationInfo &left_operation = operations[left.operation_index];
                const OperationInfo &right_operation = operations[right.operation_index];
                if (left_operation.job_id != right_operation.job_id) {
                    return left_operation.job_id < right_operation.job_id;
                }
                return left_operation.process_id < right_operation.process_id;
            }
        );

        for (const ScheduledOperation &scheduled_operation : operations_on_machine) {
            const OperationInfo &operation = operations[scheduled_operation.operation_index];
            schedule_items[machine_id].schedule_process.push_back({
                operation.job_id,
                operation.process_id
            });
        }
        schedule_items[machine_id].process_count =
            static_cast<int>(schedule_items[machine_id].schedule_process.size());
    }

    schedule.set_schedule_items(schedule_items);
    const std::vector<std::string> effective_job_list = BuildEffectiveJobList(jobList, jobs);
    std::vector<Schedule_item> graph_items = schedule.get_schedule_items();
    ScheduleItemsToGraph(
        schedule,
        graph_items,
        jobs,
        effective_job_list,
        true,
        graph_machines
    );
    return schedule;
}

Schedule RepairCoptMachineSequence(
    const Schedule &schedule,
    const std::vector<std::string> &released_processes,
    const std::vector<std::string> &jobList,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    const SearchDeadline deadline
) {
    if (released_processes.empty() || SearchDeadlineReached(deadline)) return schedule;
    struct FixedOperation {
        Schedule_process process;
        int machine_index;
        int duration;
        int start;
    };
    const auto &original_items = schedule.get_schedule_items();
    std::unordered_map<std::string, int> old_starts;
    for (size_t index = 0; index < schedule.get_processList().size(); ++index) {
        std::string key = schedule.get_processList()[index];
        if (key.rfind("job", 0) == 0) key.erase(0, 3);
        old_starts[key] = schedule.get_start_time()[index];
    }
    std::vector<FixedOperation> operations;
    std::unordered_map<std::string, int> indices;
    std::vector<std::vector<int>> machine_operations(original_items.size());
    for (int machine = 0; machine < static_cast<int>(original_items.size()); ++machine) {
        for (const auto &process : original_items[machine].schedule_process) {
            const std::string key = std::to_string(process.job_id) + "-" +
                                    std::to_string(process.process_id);
            const int duration = GetProcessTime(jobs, original_items[machine].machine_id,
                                               process.job_id, process.process_id);
            if (duration <= 0 || !old_starts.contains(key)) return schedule;
            indices[key] = static_cast<int>(operations.size());
            machine_operations[machine].push_back(indices[key]);
            operations.push_back({process, machine, duration, old_starts.at(key)});
        }
    }
    std::vector<bool> released(operations.size(), false);
    int release_count = 0;
    for (std::string key : released_processes) {
        if (key.rfind("job", 0) == 0) key.erase(0, 3);
        const auto found = indices.find(key);
        if (found != indices.end() && !released[found->second]) {
            released[found->second] = true;
            if (++release_count >= 16) break;
        }
    }
    if (release_count == 0) return schedule;
    int fixed_assignment_lower_bound = 0;
    for (const auto &sequence : machine_operations) {
        int workload = 0;
        for (const int index : sequence) workload += operations[index].duration;
        fixed_assignment_lower_bound = std::max(fixed_assignment_lower_bound, workload);
    }
    if (fixed_assignment_lower_bound >= schedule.get_TotalTime()) {
        std::cout << "Greedy sequence repair skipped: fixed_assignment_lower_bound="
                  << fixed_assignment_lower_bound << '\n';
        return schedule;
    }

    try {
        static thread_local std::unique_ptr<Envr> repair_environment;
        if (!repair_environment) repair_environment = std::make_unique<Envr>();
        Model model = repair_environment->CreateModel("HLS_Critical_Sequence_Repair");
        model.SetIntParam(COPT_INTPARAM_LOGGING, 0);
        model.SetIntParam(COPT_INTPARAM_THREADS, 1);
        const int horizon = schedule.get_TotalTime();
        std::vector<Var> starts;
        starts.reserve(operations.size());
        for (const auto &operation : operations) {
            starts.push_back(model.AddVar(0.0, horizon - operation.duration,
                                         0.0, COPT_CONTINUOUS));
            model.SetMipStart(starts.back(), operation.start);
        }
        Var makespan = model.AddVar(0.0, horizon, 0.0, COPT_CONTINUOUS);
        model.SetMipStart(makespan, horizon);
        for (int index = 0; index < static_cast<int>(operations.size()); ++index) {
            model.AddConstr(makespan >= starts[index] + operations[index].duration);
        }
        for (const auto &job : jobs) {
            int previous = -1;
            for (const auto &process : job.get_job_process()) {
                const std::string key = std::to_string(job.get_job_id()) + "-" +
                    std::to_string(process.process_item.front().process_id);
                if (!indices.contains(key)) return schedule;
                const int current = indices.at(key);
                if (previous >= 0) {
                    model.AddConstr(starts[current] >= starts[previous] +
                                    operations[previous].duration);
                }
                previous = current;
            }
        }
        for (const auto &sequence : machine_operations) {
            int previous_fixed = -1;
            for (const int index : sequence) {
                if (!released[index]) {
                    if (previous_fixed >= 0) {
                        model.AddConstr(starts[index] >= starts[previous_fixed] +
                                        operations[previous_fixed].duration);
                    }
                    previous_fixed = index;
                }
            }
            for (int left = 0; left < static_cast<int>(sequence.size()); ++left) {
                for (int right = left + 1; right < static_cast<int>(sequence.size()); ++right) {
                    const int first = sequence[left], second = sequence[right];
                    if ((!released[first] && !released[second]) ||
                        operations[first].process.job_id == operations[second].process.job_id) {
                        continue;
                    }
                    Var forward = model.AddVar(0.0, 1.0, 0.0, COPT_BINARY);
                    model.SetMipStart(forward, 1.0);
                    model.AddConstr(starts[second] - starts[first] -
                                    operations[first].duration + horizon * (1.0 - forward) >= 0.0);
                    model.AddConstr(starts[first] - starts[second] -
                                    operations[second].duration + horizon * forward >= 0.0);
                }
            }
        }
        model.SetObjective(makespan, COPT_MINIMIZE);
        model.LoadMipStart();
        const double remaining = std::chrono::duration<double>(
            deadline - SearchClock::now()).count();
        if (remaining < 0.05) return schedule;
        model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, std::min(2.0, remaining));
        model.Solve();
        if (!model.GetIntAttr(COPT_INTATTR_HASSOL)) return schedule;
        std::vector<Schedule_item> candidate_items = original_items;
        for (int machine = 0; machine < static_cast<int>(machine_operations.size()); ++machine) {
            auto sequence = machine_operations[machine];
            std::stable_sort(sequence.begin(), sequence.end(), [&](const int left, const int right) {
                return starts[left].Get(COPT_DBLINFO_VALUE) < starts[right].Get(COPT_DBLINFO_VALUE);
            });
            auto &processes = candidate_items[machine].schedule_process;
            processes.clear();
            for (const int index : sequence) processes.push_back(operations[index].process);
        }
        int sparse_makespan = 0;
        if (!TryCalculateScheduleItemsMakespan(candidate_items, jobs, jobList, sparse_makespan) ||
            sparse_makespan >= horizon) return schedule;
        Schedule candidate = schedule;
        candidate.set_schedule_items(candidate_items);
        ScheduleItemsToGraph(candidate, candidate_items, jobs, jobList, true, machines);
        if (candidate.get_TotalTime() != sparse_makespan) {
            throw std::runtime_error("COPT sequence repair sparse/dense disagreement");
        }
        CalculateTotalFailureRate(candidate, machines);
        std::cout << "Greedy sequence repair improvement: " << horizon << " -> "
                  << candidate.get_TotalTime() << ", released=" << release_count << '\n';
        return candidate;
    } catch (const std::exception &exception) {
        std::cerr << "Greedy sequence repair unavailable: " << exception.what() << '\n';
        return schedule;
    }
}

Schedule RepairCoptFlexibleNeighborhood(
    const Schedule &schedule,
    const std::vector<std::string> &released_processes,
    const std::vector<std::string> &jobList,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    const SearchDeadline deadline,
    const int release_override,
    const double seconds_override
) {
    if (released_processes.empty() || SearchDeadlineReached(deadline)) return schedule;
    int release_limit = 24;
    double repair_seconds = 3.0;
    try {
        if (const char *value = std::getenv("HLS_GREEDY_RELEASE_COUNT")) release_limit = std::clamp(std::stoi(value), 8, 48);
        if (const char *value = std::getenv("HLS_GREEDY_REPAIR_SECONDS")) {
            const double parsed = std::stod(value);
            if (std::isfinite(parsed)) repair_seconds = std::clamp(parsed, 0.1, 10.0);
        }
    } catch (const std::exception &) {
        release_limit = 24;
        repair_seconds = 3.0;
    }
    if (release_override > 0) release_limit = std::clamp(release_override, 8, 48);
    if (seconds_override > 0.0 && std::isfinite(seconds_override)) repair_seconds = std::clamp(seconds_override, 0.1, 10.0);
    struct RepairOperation {
        Schedule_process process;
        int machine;
        int duration;
        int start;
    };
    const auto &original_items = schedule.get_schedule_items();
    std::unordered_map<std::string, int> original_starts;
    for (size_t index = 0; index < schedule.get_processList().size(); ++index) {
        std::string key = schedule.get_processList()[index];
        if (key.rfind("job", 0) == 0) key.erase(0, 3);
        original_starts[key] = schedule.get_start_time()[index];
    }
    std::vector<RepairOperation> operations;
    std::unordered_map<std::string, int> indices;
    std::vector<std::vector<int>> original_sequences(original_items.size());
    std::unordered_map<int, int> machine_indices;
    for (int machine = 0; machine < static_cast<int>(original_items.size()); ++machine) {
        machine_indices[original_items[machine].machine_id] = machine;
        for (const auto &process : original_items[machine].schedule_process) {
            const std::string key = std::to_string(process.job_id) + "-" + std::to_string(process.process_id);
            const int duration = GetProcessTime(jobs, original_items[machine].machine_id,
                                               process.job_id, process.process_id);
            if (duration <= 0 || !original_starts.contains(key)) return schedule;
            const int index = static_cast<int>(operations.size());
            indices[key] = index;
            original_sequences[machine].push_back(index);
            operations.push_back({process, machine, duration, original_starts.at(key)});
        }
    }
    std::vector<bool> released(operations.size(), false);
    int release_count = 0;
    for (std::string key : released_processes) {
        if (key.rfind("job", 0) == 0) key.erase(0, 3);
        const auto found = indices.find(key);
        if (found != indices.end() && !released[found->second]) {
            released[found->second] = true;
            if (++release_count >= release_limit) break;
        }
    }
    if (release_count == 0) return schedule;
    try {
        static thread_local std::unique_ptr<Envr> repair_environment;
        if (!repair_environment) repair_environment = std::make_unique<Envr>();
        Model model = repair_environment->CreateModel("HLS_Greedy_Flexible_Neighborhood");
        model.SetIntParam(COPT_INTPARAM_LOGGING, 0);
        model.SetIntParam(COPT_INTPARAM_THREADS, 1);
        const int horizon = schedule.get_TotalTime();
        std::vector<std::vector<int>> option_machines(operations.size()), option_durations(operations.size());
        for (const auto &job : jobs) {
            for (const auto &process : job.get_job_process()) {
                if (process.process_item.empty()) return schedule;
                const std::string key = std::to_string(job.get_job_id()) + "-" +
                    std::to_string(process.process_item.front().process_id);
                if (!indices.contains(key)) return schedule;
                const int index = indices.at(key);
                for (const auto &option : process.process_item) {
                    if (!machine_indices.contains(option.machine_id)) return schedule;
                    const int machine = machine_indices.at(option.machine_id);
                    if (released[index] || machine == operations[index].machine) {
                        option_machines[index].push_back(machine);
                        option_durations[index].push_back(option.process_time);
                    }
                }
            }
        }
        // The fixed subsequences and job precedence form a DAG. Its minimum
        // durations give valid head/tail bounds even when released operations
        // change machines, and considerably tighten the disjunctive model.
        const int operation_count = static_cast<int>(operations.size());
        std::vector<int> minimum_durations(operation_count), heads(operation_count, 0), tails(operation_count, 0);
        std::vector<int> in_degree(operation_count, 0), topological_order;
        std::vector<std::vector<int>> successors(operation_count);
        for (int index = 0; index < operation_count; ++index) {
            if (option_durations[index].empty()) return schedule;
            minimum_durations[index] = *std::min_element(option_durations[index].begin(), option_durations[index].end());
        }
        auto add_precedence = [&](const int first, const int second) {
            successors[first].push_back(second);
            ++in_degree[second];
        };
        for (const auto &job : jobs) {
            int previous = -1;
            for (const auto &process : job.get_job_process()) {
                const int current = indices.at(std::to_string(job.get_job_id()) + "-" +
                    std::to_string(process.process_item.front().process_id));
                if (previous >= 0) add_precedence(previous, current);
                previous = current;
            }
        }
        for (const auto &sequence : original_sequences) {
            int previous = -1;
            for (const int index : sequence) {
                if (released[index]) continue;
                if (previous >= 0) add_precedence(previous, index);
                previous = index;
            }
        }
        std::queue<int> ready;
        for (int index = 0; index < operation_count; ++index) if (in_degree[index] == 0) ready.push(index);
        while (!ready.empty()) {
            const int first = ready.front();
            ready.pop();
            topological_order.push_back(first);
            for (const int second : successors[first]) {
                heads[second] = std::max(heads[second], heads[first] + minimum_durations[first]);
                if (--in_degree[second] == 0) ready.push(second);
            }
        }
        if (static_cast<int>(topological_order.size()) != operation_count) return schedule;
        for (auto it = topological_order.rbegin(); it != topological_order.rend(); ++it) {
            for (const int second : successors[*it]) {
                tails[*it] = std::max(tails[*it], minimum_durations[second] + tails[second]);
            }
        }
        std::vector<Var> starts;
        std::vector<Expr> durations;
        std::vector<std::vector<Var>> assignments(operations.size());
        std::vector<std::vector<std::pair<int, int>>> possible_sequences(original_items.size());
        for (int index = 0; index < static_cast<int>(operations.size()); ++index) {
            if (SearchDeadlineReached(deadline)) return schedule;
            if (option_machines[index].empty()) return schedule;
            const int upper = horizon - tails[index] - minimum_durations[index];
            if (heads[index] > upper) return schedule;
            starts.push_back(model.AddVar(heads[index], upper, 0.0, COPT_CONTINUOUS));
            model.SetMipStart(starts.back(), operations[index].start);
            Expr selected;
            Expr duration;
            for (int option = 0; option < static_cast<int>(option_machines[index].size()); ++option) {
                Var assignment = model.AddVar(0.0, 1.0, 0.0, COPT_BINARY);
                assignments[index].push_back(assignment);
                model.SetMipStart(assignment, option_machines[index][option] == operations[index].machine ? 1.0 : 0.0);
                selected += assignment;
                duration += option_durations[index][option] * assignment;
                possible_sequences[option_machines[index][option]].push_back({index, option});
            }
            model.AddConstr(selected == 1.0);
            durations.push_back(duration);
        }
        Var makespan = model.AddVar(0.0, horizon, 0.0, COPT_CONTINUOUS);
        model.SetMipStart(makespan, horizon);
        for (int index = 0; index < static_cast<int>(operations.size()); ++index) {
            model.AddConstr(makespan >= starts[index] + durations[index] + tails[index]);
        }
        for (const auto &job : jobs) {
            int previous = -1;
            for (const auto &process : job.get_job_process()) {
                const std::string key = std::to_string(job.get_job_id()) + "-" +
                    std::to_string(process.process_item.front().process_id);
                const int current = indices.at(key);
                if (previous >= 0) model.AddConstr(starts[current] >= starts[previous] + durations[previous]);
                previous = current;
            }
        }
        for (const auto &sequence : original_sequences) {
            int previous = -1;
            for (const int index : sequence) {
                if (released[index]) continue;
                if (previous >= 0) model.AddConstr(starts[index] >= starts[previous] + durations[previous]);
                previous = index;
            }
        }
        for (const auto &sequence : possible_sequences) {
            Expr workload;
            for (const auto &[index, option] : sequence) {
                workload += option_durations[index][option] * assignments[index][option];
            }
            model.AddConstr(makespan >= workload);
            for (int left = 0; left < static_cast<int>(sequence.size()); ++left) {
                if (SearchDeadlineReached(deadline)) return schedule;
                for (int right = left + 1; right < static_cast<int>(sequence.size()); ++right) {
                    const auto [first, first_option] = sequence[left];
                    const auto [second, second_option] = sequence[right];
                    if ((!released[first] && !released[second]) ||
                        operations[first].process.job_id == operations[second].process.job_id) continue;
                    if (horizon - tails[first] <= heads[second] ||
                        horizon - tails[second] <= heads[first]) continue;
                    Var forward = model.AddVar(0.0, 1.0, 0.0, COPT_BINARY);
                    model.SetMipStart(forward, operations[first].start <= operations[second].start ? 1.0 : 0.0);
                    const Var first_selected = assignments[first][first_option];
                    const Var second_selected = assignments[second][second_option];
                    model.AddConstr(starts[second] - starts[first] - durations[first] +
                                    (horizon - tails[first] - heads[second]) *
                                        (3.0 - forward - first_selected - second_selected) >= 0.0);
                    model.AddConstr(starts[first] - starts[second] - durations[second] +
                                    (horizon - tails[second] - heads[first]) *
                                        (2.0 + forward - first_selected - second_selected) >= 0.0);
                }
            }
        }
        model.SetObjective(makespan, COPT_MINIMIZE);
        model.LoadMipStart();
        const double remaining = std::chrono::duration<double>(deadline - SearchClock::now()).count();
        if (remaining < 0.05) return schedule;
        model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, std::min(repair_seconds, remaining));
        model.Solve();
        if (!model.GetIntAttr(COPT_INTATTR_HASSOL)) return schedule;
        std::vector<std::vector<std::pair<double, int>>> scheduled(original_items.size());
        for (int index = 0; index < static_cast<int>(operations.size()); ++index) {
            int selected = -1;
            for (int option = 0; option < static_cast<int>(assignments[index].size()); ++option) {
                if (assignments[index][option].Get(COPT_DBLINFO_VALUE) > 0.5) {
                    selected = option_machines[index][option];
                    break;
                }
            }
            if (selected < 0) return schedule;
            scheduled[selected].push_back({starts[index].Get(COPT_DBLINFO_VALUE), index});
        }
        auto items = original_items;
        for (int machine = 0; machine < static_cast<int>(items.size()); ++machine) {
            std::sort(scheduled[machine].begin(), scheduled[machine].end());
            items[machine].schedule_process.clear();
            for (const auto &[start, index] : scheduled[machine]) {
                items[machine].schedule_process.push_back(operations[index].process);
            }
            items[machine].process_count = static_cast<int>(items[machine].schedule_process.size());
        }
        int sparse = 0;
        if (!TryCalculateScheduleItemsMakespan(items, jobs, jobList, sparse) || sparse >= horizon) return schedule;
        Schedule result = schedule;
        result.set_schedule_items(items);
        ScheduleItemsToGraph(result, items, jobs, jobList, true, machines);
        if (result.get_TotalTime() != sparse) throw std::runtime_error("Flexible repair sparse/dense disagreement");
        CalculateTotalFailureRate(result, machines);
        std::cout << "Greedy flexible repair improvement: " << horizon << " -> " << sparse
                  << ", released=" << release_count << '\n';
        return result;
    } catch (const std::exception &error) {
        std::cerr << "Greedy flexible repair unavailable: " << error.what() << '\n';
        return schedule;
    }
}
