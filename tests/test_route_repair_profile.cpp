#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "RouteRepairProfile.h"

namespace {

void Require(const bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<Job> BuildProblem(
    const int job_count,
    const int operations_per_job,
    const int options_per_operation
) {
    std::vector<Job> jobs(job_count);
    for (int job_index = 0; job_index < job_count; ++job_index) {
        std::vector<Job_process> processes;
        for (int process_index = 0;
             process_index < operations_per_job;
             ++process_index) {
            Job_process process;
            process.machine_count = options_per_operation;
            for (int option = 0; option < options_per_operation; ++option) {
                process.process_item.push_back({process_index, option, 10 + option});
            }
            processes.push_back(std::move(process));
        }
        jobs[job_index].set_job_id(job_index);
        jobs[job_index].set_job_name("job" + std::to_string(job_index));
        jobs[job_index].set_process_count(operations_per_job);
        jobs[job_index].set_job_process(processes);
    }
    return jobs;
}

}  // namespace

int main() {
    Require(IsLongRouteFlexibleProblem(BuildProblem(10, 18, 2)),
            "long flexible problem must match the base profile");
    Require(!IsLongRouteFlexibleProblem(BuildProblem(10, 17, 4)),
            "short routes must retain the original neighborhood");
    Require(!IsLongRouteFlexibleProblem(BuildProblem(10, 20, 1)),
            "low flexibility must retain the original neighborhood");
    Require(IsRouteCoupledRepairProblem(BuildProblem(15, 18, 2)),
            "medium-scale long flexible routes must enable route repair");
    Require(!IsRouteCoupledRepairProblem(BuildProblem(10, 18, 2)),
            "small job sets must retain standard repair");
    Require(IsRouteCoupledRepairProblem(BuildProblem(20, 18, 3)),
            "large moderately flexible routes must enable route repair");
    Require(!IsRouteCoupledRepairProblem(BuildProblem(20, 18, 4)),
            "very high-flexibility large jobs must retain standard repair");
}
