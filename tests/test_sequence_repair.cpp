#include <iostream>
#include <stdexcept>
#include <vector>

#include "CoptInitialSolution.h"
#include "DataProc.h"

int main() {
    std::vector<Job> jobs;
    std::vector<Machine> machines;
    std::vector<Order> orders;
    std::vector<std::string> job_list;
    Init(machines, jobs, orders, "tests/data/decode_idle_gap.fjs");
    OrderToJobList(orders, job_list);
    Schedule original = GenerateInitialSolution(job_list, jobs, machines);
    if (original.get_TotalTime() != 15) throw std::runtime_error("Fixture must finish at 15");
    const Schedule repaired = RepairCoptMachineSequence(
        original, {"0-1", "1-0"}, job_list, jobs, machines,
        SearchClock::now() + std::chrono::seconds(5));
    if (repaired.get_TotalTime() != 12) throw std::runtime_error("Repair must eliminate idle gap");
    int sparse_makespan = 0;
    if (!TryCalculateScheduleItemsMakespan(repaired.get_schedule_items(), jobs, job_list,
                                          sparse_makespan) || sparse_makespan != 12) {
        throw std::runtime_error("Repair is not precedence/resource feasible");
    }
    for (size_t machine = 0; machine < original.get_schedule_items().size(); ++machine) {
        if (original.get_schedule_items()[machine].process_count !=
            repaired.get_schedule_items()[machine].process_count) {
            throw std::runtime_error("Sequence repair must retain machine assignments");
        }
    }
    std::cout << "Sequence repair verified: 15 -> 12 with fixed machine assignments.\n";
}
