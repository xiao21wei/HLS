#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "DataProc.h"
#include "GreedySearch.h"
#include "GreedyRlPolicy.h"
#include "GuidedSearch.h"
#include "RandomSearch.h"
#include "TabuSearch.h"
#include "TabuRlPolicy.h"

int main() {
    std::vector<Job> jobs;
    std::vector<Machine> machines;
    std::vector<Order> orders;
    std::vector<std::string> job_list;
    Init(machines, jobs, orders, "tests/data/decode_flexible.fjs");
    OrderToJobList(orders, job_list);
    const Schedule initial = GenerateInitialSolution(job_list, jobs, machines);
    setenv("HLS_GREEDY_RL", "0", 1);
    setenv("HLS_TABU_RL", "0", 1);
    ConfigureGreedyRlPolicyFromEnvironment(17);
    ConfigureTabuRlPolicyFromEnvironment(19);

    auto run_phase = [&](const unsigned int seed, const bool reset) {
        if (reset) {
            ResetGeneticSearchState(seed ^ 1U);
            ResetTabuSearchState(seed ^ 2U);
            ResetGuidedSearchState(seed ^ 3U);
            ResetGreedySearchState();
            SetGreedyRlSamplingSeed(seed ^ 4U);
            SetTabuRlSamplingSeed(seed ^ 5U);
        }
        Schedule current = GreedySearch(initial, jobs, job_list, 3, machines);
        current = TabuSearch(current, jobs, job_list, 7, 15, machines);
        current = AggressiveSearch(current, jobs, machines, job_list, 8, 12);
        current = GuidedInsertionSearch(current, {initial, current}, jobs,
                                        job_list, machines, 15);
        int sparse_makespan = 0;
        if (!TryCalculateScheduleItemsMakespan(current.get_schedule_items(), jobs,
                                              job_list, sparse_makespan) ||
            sparse_makespan != current.get_TotalTime()) {
            throw std::runtime_error("Restart phase returned an invalid schedule");
        }
        return current;
    };

    const Schedule first = run_phase(97U, true);
    for (int phase = 0; phase < 5; ++phase) run_phase(phase + 100U, false);
    const Schedule restarted = run_phase(97U, true);
    if (ScheduleToString(first) != ScheduleToString(restarted) ||
        first.get_TotalTime() != restarted.get_TotalTime()) {
        throw std::runtime_error("Restart retained state from earlier phases");
    }
    std::cout << "Greedy/GA/Tabu/Guided restart isolation verified after five prior phases.\n";
}
