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
    Init(machines, jobs, orders, "tests/data/flexible_repair.fjs");
    OrderToJobList(orders, job_list);
    Schedule original;
    original.set_machine_count(2);
    std::vector<Schedule_item> items(2);
    items[0].machine_id = 0;
    items[0].process_count = 1;
    items[0].schedule_process = {{0, 0}};
    items[1].machine_id = 1;
    items[1].process_count = 1;
    items[1].schedule_process = {{1, 0}};
    original.set_schedule_items(items);
    ScheduleItemsToGraph(original, items, jobs, job_list, true, machines);
    if (original.get_TotalTime() != 10) throw std::runtime_error("Expected makespan 10");
    const Schedule repaired = RepairCoptFlexibleNeighborhood(
        original, {"0-0"}, job_list, jobs, machines,
        SearchClock::now() + std::chrono::seconds(5));
    if (repaired.get_TotalTime() != 5) throw std::runtime_error("Must move released operation to machine 2");
    if (repaired.get_schedule_items()[1].process_count != 2 ||
        repaired.get_schedule_items()[0].process_count != 0) throw std::runtime_error("Wrong machine counts");
    int sparse = 0;
    if (!TryCalculateScheduleItemsMakespan(repaired.get_schedule_items(), jobs, job_list, sparse) || sparse != 5) {
        throw std::runtime_error("Repair is not machine/precedence feasible");
    }
    const Schedule expired = RepairCoptFlexibleNeighborhood(
        original, {"0-0"}, job_list, jobs, machines, SearchClock::now());
    if (expired.get_TotalTime() != 10) throw std::runtime_error("Expired deadline must preserve input");
    std::cout << "Flexible repair: 10 -> 5; immutable fixed operation; valid counts and deadline.\n";
}
