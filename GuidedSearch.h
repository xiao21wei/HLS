#ifndef GUIDEDSEARCH_H
#define GUIDEDSEARCH_H

#include <vector>

#include "Job.h"
#include "Machine.h"
#include "Schedule.h"
#include "SearchDeadline.h"

void SetGuidedSearchSeed(unsigned int random_seed);

Schedule GuidedInsertionSearch(
    const Schedule &schedule,
    const std::vector<Schedule> &seed_schedules,
    const std::vector<Job> &jobs,
    const std::vector<std::string> &job_list,
    std::vector<Machine> &machines,
    int iteration_count,
    SearchDeadline deadline = SearchDeadline::max()
);

#endif  // GUIDEDSEARCH_H
