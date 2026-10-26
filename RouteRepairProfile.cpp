#include "RouteRepairProfile.h"

bool IsLongRouteFlexibleProblem(const std::vector<Job> &jobs) {
    if (jobs.empty()) {
        return false;
    }
    long long operation_count = 0;
    long long option_count = 0;
    for (const auto &job : jobs) {
        for (const auto &process : job.get_job_process()) {
            ++operation_count;
            option_count += static_cast<long long>(process.process_item.size());
        }
    }
    return operation_count >= static_cast<long long>(jobs.size()) * 18 &&
           option_count >= operation_count * 2;
}

bool IsRouteCoupledRepairProblem(const std::vector<Job> &jobs) {
    if (!IsLongRouteFlexibleProblem(jobs)) {
        return false;
    }
    if (jobs.size() >= 12 && jobs.size() <= 16) {
        return true;
    }
    long long operation_count = 0;
    long long option_count = 0;
    for (const auto &job : jobs) {
        for (const auto &process : job.get_job_process()) {
            ++operation_count;
            option_count += static_cast<long long>(process.process_item.size());
        }
    }
    return jobs.size() >= 18 && option_count < operation_count * 4;
}
