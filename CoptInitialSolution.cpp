#include "CoptInitialSolution.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
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
