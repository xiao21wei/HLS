#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataProc.h"
#include "RandomSearch.h"

void Require(const bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckFeasibility(const Schedule &schedule, const std::vector<Job> &jobs,
                      const std::vector<std::string> &job_list) {
    int sparse_makespan = 0;
    Require(TryCalculateScheduleItemsMakespan(schedule.get_schedule_items(), jobs,
                                             job_list, sparse_makespan),
            "Decoded schedule is not feasible");
    Require(sparse_makespan == schedule.get_TotalTime(), "Makespan mismatch");
    std::unordered_map<std::string, int> starts;
    for (size_t i = 0; i < schedule.get_processList().size(); ++i) {
        starts[schedule.get_processList()[i]] = schedule.get_start_time()[i];
    }
    std::unordered_map<std::string, int> finishes;
    for (const auto &item : schedule.get_schedule_items()) {
        int previous_finish = 0;
        Require(item.process_count == static_cast<int>(item.schedule_process.size()),
                "Incorrect machine operation count");
        for (const auto &operation : item.schedule_process) {
            const std::string key = std::to_string(operation.job_id) + "-" +
                                    std::to_string(operation.process_id);
            const int duration = GetProcessTime(jobs, item.machine_id,
                                               operation.job_id, operation.process_id);
            Require(duration > 0, "Incompatible machine assignment");
            const auto start_it = starts.find(key);
            const auto prefixed_it = starts.find("job" + key);
            Require(start_it != starts.end() || prefixed_it != starts.end(),
                    "Missing operation start time");
            const int start = start_it != starts.end() ? start_it->second : prefixed_it->second;
            Require(start >= previous_finish, "Machine overlap");
            previous_finish = start + duration;
            finishes[key] = previous_finish;
        }
    }
    for (const auto &job : jobs) {
        for (int process = 1; process < job.get_process_count(); ++process) {
            const std::string key = std::to_string(job.get_job_id()) + "-" +
                                    std::to_string(process);
            const int start = starts.contains(key) ? starts.at(key) : starts.at("job" + key);
            Require(start >= finishes.at(std::to_string(job.get_job_id()) + "-" +
                                          std::to_string(process - 1)),
                    "Job precedence violation");
        }
    }
}

int main() {
    std::vector<Job> jobs;
    std::vector<Machine> machines;
    std::vector<Order> orders;
    std::vector<std::string> job_list;
    Init(machines, jobs, orders, "tests/data/decode_idle_gap.fjs");
    OrderToJobList(orders, job_list);
    Schedule base = GenerateInitialSolution(job_list, jobs, machines);
    setenv("HLS_GA_ACTIVE_DECODE", "0", 1);
    const Schedule append = DecodeSchedule({0, 0, 0}, {0, 0, 1}, jobs,
                                           job_list, base, machines);
    Require(append.get_TotalTime() == 15, "Append fixture must finish at 15");
    setenv("HLS_GA_ACTIVE_DECODE", "1", 1);
    const Schedule active = DecodeSchedule({0, 0, 0}, {0, 0, 1}, jobs,
                                           job_list, base, machines);
    Require(active.get_TotalTime() == 12, "Idle-gap fixture must finish at 12");
    CheckFeasibility(active, jobs, job_list);

    jobs.clear(); machines.clear(); orders.clear(); job_list.clear();
    Init(machines, jobs, orders, "tests/data/decode_flexible.fjs");
    OrderToJobList(orders, job_list);
    base = GenerateInitialSolution(job_list, jobs, machines);
    int checked = 0;
    for (int bits = 0; bits < 64; ++bits) {
        std::vector<int> machine_code(6);
        for (int gene = 0; gene < 6; ++gene) machine_code[gene] = (bits >> gene) & 1;
        std::vector<int> order{0, 0, 1, 1, 2, 2};
        do {
            Schedule decoded = DecodeSchedule(machine_code, order, jobs, job_list, base, machines);
            CheckFeasibility(decoded, jobs, job_list);
            std::vector<int> encoded_machines, encoded_order;
            EncodeSchedule(decoded, jobs, job_list, encoded_machines, encoded_order);
            const Schedule roundtrip = DecodeSchedule(encoded_machines, encoded_order,
                                                      jobs, job_list, base, machines);
            Require(roundtrip.get_TotalTime() <= decoded.get_TotalTime(),
                    "Encode/decode roundtrip worsens active schedule");
            ++checked;
        } while (std::next_permutation(order.begin(), order.end()));
    }
    std::cout << "Active decoder verified: idle-gap improvement and " << checked
              << " flexible chromosomes, precedence, resource use, roundtrip.\n";
}
