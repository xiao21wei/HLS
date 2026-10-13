#ifndef COPT_INITIAL_SOLUTION_H
#define COPT_INITIAL_SOLUTION_H

#include <string>
#include <vector>

#include "Job.h"
#include "Machine.h"
#include "Schedule.h"
#include "SearchDeadline.h"

// Generate a feasible FJSP schedule with COPT and use it as the HLS start point.
// copt_time_limit_seconds is the COPT MIP solve time limit, in seconds.
Schedule GenerateCoptInitialSolution(
    const std::vector<std::string> &jobList,
    const std::vector<Job> &jobs,
    const std::vector<Machine> &machines,
    double copt_time_limit_seconds
);

Schedule RepairCoptMachineSequence(
    const Schedule &schedule,
    const std::vector<std::string> &released_processes,
    const std::vector<std::string> &jobList,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    SearchDeadline deadline
);

// A bounded Greedy neighborhood: release a small set of operations while
// preserving the relative machine order of all other operations.
Schedule RepairCoptFlexibleNeighborhood(
    const Schedule &schedule,
    const std::vector<std::string> &released_processes,
    const std::vector<std::string> &jobList,
    const std::vector<Job> &jobs,
    std::vector<Machine> &machines,
    SearchDeadline deadline,
    int release_override = 0,
    double seconds_override = 0.0
);

#endif // COPT_INITIAL_SOLUTION_H
