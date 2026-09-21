#ifndef COPT_INITIAL_SOLUTION_H
#define COPT_INITIAL_SOLUTION_H

#include <string>
#include <vector>

#include "Job.h"
#include "Machine.h"
#include "Schedule.h"

// Generate a feasible FJSP schedule with COPT and use it as the HLS start point.
// copt_time_limit_seconds is the COPT MIP solve time limit, in seconds.
Schedule GenerateCoptInitialSolution(
    const std::vector<std::string> &jobList,
    const std::vector<Job> &jobs,
    const std::vector<Machine> &machines,
    double copt_time_limit_seconds
);

#endif // COPT_INITIAL_SOLUTION_H
