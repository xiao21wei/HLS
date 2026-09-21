//
// Created by luopw on 24-12-1.
//

#include "DataProc.h"
#include "Schedule.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>

namespace {

std::string BuildProcessKey(const int job_id, const int process_id) {
    return std::to_string(job_id) + "-" + std::to_string(process_id);
}

struct ProcessRouteInfo {
    int index = -1;
    int machine_id = -1;
};

std::unordered_map<std::string, const Job *> BuildJobNameMap(const std::vector<Job> &jobs) {
    std::unordered_map<std::string, const Job *> job_name_map;
    job_name_map.reserve(jobs.size());
    for (const auto &job : jobs) {
        job_name_map[job.get_job_name()] = &job;
    }
    return job_name_map;
}

std::unordered_map<int, const Job *> BuildJobIdMap(const std::vector<Job> &jobs) {
    std::unordered_map<int, const Job *> job_id_map;
    job_id_map.reserve(jobs.size());
    for (const auto &job : jobs) {
        job_id_map[job.get_job_id()] = &job;
    }
    return job_id_map;
}

std::unordered_map<long long, int> BuildProcessTimeMap(const std::vector<Job> &jobs) {
    std::unordered_map<long long, int> process_time_map;
    for (const auto &job : jobs) {
        const auto job_processes = job.get_job_process();
        for (const auto &job_process : job_processes) {
            for (const auto &item : job_process.process_item) {
                const long long key =
                    (static_cast<long long>(job.get_job_id()) << 32) ^
                    (static_cast<long long>(item.process_id) << 16) ^
                    static_cast<unsigned int>(item.machine_id);
                process_time_map[key] = item.process_time;
            }
        }
    }
    return process_time_map;
}

long long BuildProcessTimeKey(const int job_id, const int process_id, const int machine_id) {
    return (static_cast<long long>(job_id) << 32) ^
           (static_cast<long long>(process_id) << 16) ^
           static_cast<unsigned int>(machine_id);
}

}  // namespace


void Init(std::vector<Machine> &machines, std::vector<Job> &jobs, std::vector<Order> &orders, std::string file_path) {
    // // 重定向输入流为当前目录下的input.txt文件,使得使用std::cin读取文件内容，使用std::cout输出到控制台
    std::string inputFilePath = file_path;
    // std::cout << "Input file path: " << inputFilePath << std::endl;
    if (inputFilePath.empty()) {
        std::cerr << "Error: Input file path is empty." << std::endl;
        return;
    }
    std::ifstream in(inputFilePath);
    std::cin.rdbuf(in.rdbuf());
    if (!in) {
        std::cerr << "Error: Cannot open input file: "<< inputFilePath << std::endl;
        return;
    }

    int machineNum = 0;
    int jobNum = 0;
    // The third Brandimarte header field is the average operation count and
    // may be fractional (for example, 3.5 or 1.5).
    double averageProcessCount = 0.0;
    if (!(std::cin >> jobNum >> machineNum >> averageProcessCount)) {
        std::cerr << "Error: Invalid FJSP instance header: " << inputFilePath << std::endl;
        return;
    }
    int max_machine_id = -1;
    bool has_machine_zero = false;
    // std::cout << "jobNum: " << jobNum << ", machineNum: " << machineNum << std::endl;
    for (int i = 0; i < jobNum; i++) {
        Job jobItem;
        int processCount;
        std::cin >> processCount;
        jobItem.set_process_count(processCount);
        jobItem.set_job_id(i);
        jobItem.set_job_name("job" + std::to_string(i));
        std::vector<Job_process> jobProcess;
        for (int j=0; j<processCount; j++) {
            Job_process jobProcessItem;
            int machineCount;
            std::cin >> machineCount;
            jobProcessItem.machine_count = machineCount;
            for (int k=0; k<machineCount; k++) {
                Process_item processItem;
                std::cin >> processItem.machine_id >> processItem.process_time;
                processItem.process_id = j;
                max_machine_id = std::max(max_machine_id, processItem.machine_id);
                has_machine_zero = has_machine_zero || processItem.machine_id == 0;
                jobProcessItem.process_item.push_back(processItem);
            }
            jobProcess.push_back(jobProcessItem);
        }
        jobItem.set_job_process(jobProcess);
        jobs.push_back(jobItem);
    }

    // Kacem instances use machine ids 1..machineNum, while the HLS data
    // structures use 0..machineNum-1. This condition is unambiguous for a
    // valid 0-based instance because machineNum is outside its id range.
    if (!has_machine_zero && max_machine_id == machineNum) {
        for (Job &job : jobs) {
            std::vector<Job_process> normalized_processes = job.get_job_process();
            for (Job_process &job_process : normalized_processes) {
                for (Process_item &process_item : job_process.process_item) {
                    --process_item.machine_id;
                }
            }
            job.set_job_process(normalized_processes);
        }
    }

    for (int i = 0; i < machineNum; i++) {
        Machine machineItem;
        machineItem.set_machine_id(i);
        machineItem.set_machine_name("machine" + std::to_string(i));
        machineItem.set_machine_type(Machine_type::Job_shop_multi_process_machine);
        machineItem.set_count(1);
        machineItem.set_idle_count(1);
        machines.push_back(machineItem);
    }
    std::vector<Order_item> orderItems;
    for (int i = 0; i < jobNum; i++) {
        Order_item orderItemItem;
        orderItemItem.job_name = "job" + std::to_string(i);
        orderItemItem.job_count = 1;
        orderItems.push_back(orderItemItem);
    }
    Order orderItem;
    orderItem.set_order_id("order0");
    orderItem.set_order_items(orderItems);
    orders.push_back(orderItem);
}

void OrderToJobList(const std::vector<Order> &orders, std::vector<std::string> &jobList) {
    for (auto &orderItem : orders) {
        for (auto &[job_name, job_count] : orderItem.get_order_items()) {
            for (int i=0;i<job_count;i++) {
                jobList.push_back(job_name);
            }
        }
    }
}

Schedule GenerateSolution(std::string file_path, const std::vector<std::string> &jobList, const std::vector<Job> &jobs, std::vector<Machine> &machines)
{
    Schedule schedule;
    std::string inputFilePath = file_path;
    // std::cout << "Input file path: " << inputFilePath << std::endl;
    if (inputFilePath.empty())
    {
        std::cerr << "Error: Input file path is empty." << std::endl;
        return schedule;
    }
    std::ifstream in(inputFilePath);
    std::cin.rdbuf(in.rdbuf());
    if (!in)
    {
        std::cerr << "Error: Cannot open input file: " << inputFilePath << std::endl;
        return schedule;
    }
    int machineNum = 0;
    std::cin >> machineNum;
    schedule.set_machine_count(machineNum);
    schedule.set_schedule_id(0);

    std::vector<Schedule_item> temp_schedule_items;

    for (int i = 0; i < machineNum; i++)
    {
        Schedule_item scheduleItem;
        scheduleItem.machine_id = i;
        int ope_num = 0;
        std::cin >> ope_num;
        scheduleItem.process_count = ope_num;
        for (int j = 0; j < ope_num; j++)
        {
            int job_id = 0, process_id = 0;
            std::cin >> job_id >> process_id;
            Schedule_process scheduleProcess;
            scheduleProcess.job_id = job_id;
            scheduleProcess.process_id = process_id;
            scheduleItem.schedule_process.push_back(scheduleProcess);
        }
        temp_schedule_items.push_back(scheduleItem);
    }
    schedule.set_schedule_items(temp_schedule_items);
    // 输出schedule items
    // for (auto &item : temp_schedule_items) {
    //     std::cout << "Machine ID: " << item.machine_id << ", Process Count: " << item.process_count << std::endl;
    //     for (auto &process : item.schedule_process) {
    //         std::cout << "  Job ID: " << process.job_id << ", Process ID: " << process.process_id << std::endl;
    //     }
    // }
    std::vector<std::vector<int>> graph = ScheduleItemsToGraph(schedule, temp_schedule_items, jobs, jobList, true, machines);
    schedule.set_schedule_id(0);
    return schedule;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
Schedule GenerateInitialSolution(const std::vector<std::string> &jobList, const std::vector<Job> &jobs, std::vector<Machine> &machines) {
    Schedule schedule;
    const int machineNum = machines.size();
    schedule.set_machine_count(machineNum);
    auto temp_schedule_items = std::vector<Schedule_item>();
    for (int i = 0; i < machineNum; i++) {
        Schedule_item scheduleItem;
        scheduleItem.machine_id = i;
        scheduleItem.process_count = 0;
        scheduleItem.schedule_process.clear();
        temp_schedule_items.push_back(scheduleItem);
    }
    schedule.set_schedule_items(temp_schedule_items);

    std::vector<int> machine_available_time(machineNum, 0);
    std::unordered_map<int, int> job_available_time;
    job_available_time.reserve(jobs.size());
    std::vector<double> machine_failure_rate(machineNum, 0.0);
    for (const auto &machine : machines) {
        if (machine.get_machine_id() >= 0 &&
            machine.get_machine_id() < machineNum) {
            machine_failure_rate[machine.get_machine_id()] =
                machine.get_failure_rate();
        }
    }

    // Build a feasible list schedule using processing time and precedence
    // readiness.  Balancing operation counts alone can put long operations on
    // the same machine and creates a poor starting critical path.
    for (const auto &jobName : jobList) {
        const auto job = SelectJobByJobName(jobs, jobName);
        if (job.get_job_name() == "") {
            std::cout << "Error: Cannot find job by job name." << std::endl;
            return Schedule();
        }
        int &job_ready_time = job_available_time[job.get_job_id()];
        for (const auto &[machine_count, process_item] : job.get_job_process()) {
            if (process_item.empty()) {
                continue;
            }

            int best_machine_id = -1;
            int best_finish_time = INT_MAX;
            int best_processing_time = INT_MAX;
            double best_failure_rate = std::numeric_limits<double>::infinity();
            for (const auto &option : process_item) {
                if (option.machine_id < 0 || option.machine_id >= machineNum ||
                    option.process_time <= 0) {
                    continue;
                }
                const int start_time = std::max(
                    job_ready_time,
                    machine_available_time[option.machine_id]
                );
                const int finish_time = start_time + option.process_time;
                const double failure_rate =
                    machine_failure_rate[option.machine_id];
                const bool better =
                    finish_time < best_finish_time ||
                    (finish_time == best_finish_time &&
                     option.process_time < best_processing_time) ||
                    (finish_time == best_finish_time &&
                     option.process_time == best_processing_time &&
                     machine_available_time[option.machine_id] <
                         (best_machine_id < 0
                              ? INT_MAX
                              : machine_available_time[best_machine_id])) ||
                    (finish_time == best_finish_time &&
                     option.process_time == best_processing_time &&
                     best_machine_id >= 0 &&
                     failure_rate < best_failure_rate);
                if (better) {
                    best_machine_id = option.machine_id;
                    best_finish_time = finish_time;
                    best_processing_time = option.process_time;
                    best_failure_rate = failure_rate;
                }
            }
            if (best_machine_id < 0) {
                throw std::runtime_error(
                    "Initial solution contains an operation without a valid machine."
                );
            }

            schedule.AddProcess(
                best_machine_id,
                job.get_job_id(),
                process_item.front().process_id
            );
            machine_available_time[best_machine_id] = best_finish_time;
            job_ready_time = best_finish_time;
        }
    }
    std::vector<Schedule_item> temp_items = schedule.get_schedule_items();
    std::vector<std::vector<int>> graph = ScheduleItemsToGraph(schedule, temp_items, jobs, jobList, true, machines);
    schedule.set_schedule_id(0);
    return schedule;
}

Job SelectJobByJobName(const std::vector<Job> &jobs, const std::string &jobName) {
    const std::string temp = jobName.substr(0, jobName.find('-'));
    for (auto &job : jobs) {
        if (job.get_job_name() == temp) {
            return job;
        }
    }
    return Job();
}

int GetProcessTime(const std::vector<Job> &jobs, const int machine_id, const int job_id, const int process_id) {
    for (const auto &job : jobs) {
        if (job.get_job_id() == job_id) {
            for (const auto &[machine_count, process_item] : job.get_job_process()) {
                for (const auto &[temp_process_id, temp_machine_id, temp_process_time] : process_item) {
                    if (machine_id == temp_machine_id && process_id == temp_process_id) {
                        return temp_process_time;
                    }
                }
            }
        }
    }
    return 0;
}

std::vector<std::vector<int>> ScheduleItemsToGraph(Schedule &schedule, std::vector<Schedule_item> &schedule_items, const std::vector<Job> &jobs, const std::vector<std::string> &jobList, const bool flag, std::vector<Machine> &machines) {
    // std::cout << "ScheduleItemsToGraph start" << std::endl;
    std::vector<std::string> processList;  // 用于记录每个工序列表
    int scheduled_process_count = 0;
    for (const auto &schedule_item : schedule_items) {
        scheduled_process_count += schedule_item.process_count;
    }
    processList.reserve(scheduled_process_count + 2);
    processList.push_back("start");
    std::unordered_map<std::string, ProcessRouteInfo> process_route_map;
    process_route_map.reserve(scheduled_process_count);
    for (const auto &schedule_item: schedule_items) {
        for (const auto &[job_id, process_id]: schedule_item.schedule_process) {
            const std::string process_key = BuildProcessKey(job_id, process_id);
            process_route_map[process_key] = {static_cast<int>(processList.size()), schedule_item.machine_id};
            processList.push_back(process_key);
        }
    }
    processList.push_back("end");
    const int processCount = processList.size();
    const int end_index = processCount - 1;
    const auto process_time_map = BuildProcessTimeMap(jobs);
    const auto job_name_map = BuildJobNameMap(jobs);

    // std::cout << "processCount: " << processCount << std::endl;
    // for (auto process: processList) {
    //     std::cout << process << " ";
    // }
    // std::cout << std::endl;

    // 使用邻接矩阵记录工序之间的关系
    std::vector<std::vector<int>> graph(processCount, std::vector<int>(processCount, -1));

    for (const auto &schedule_item: schedule_items) {
        for (int i = 0; i < schedule_item.schedule_process.size(); ++i) {
            const int job_id = schedule_item.schedule_process[i].job_id;
            const int process_id = schedule_item.schedule_process[i].process_id;
            const int index1 = process_route_map[BuildProcessKey(job_id, process_id)].index;
            if (i == 0 && process_id == 0) {
                graph[0][index1] = 0;
            }
            if (i == schedule_item.schedule_process.size() - 1) {
                const auto process_time_it = process_time_map.find(BuildProcessTimeKey(job_id, process_id, schedule_item.machine_id));
                graph[index1][end_index] = process_time_it == process_time_map.end() ? 0 : process_time_it->second;
            }
            if (i != 0) {
                const int prev_job_id = schedule_item.schedule_process[i - 1].job_id;
                const int prev_process_id = schedule_item.schedule_process[i - 1].process_id;
                const int index2 = process_route_map[BuildProcessKey(prev_job_id, prev_process_id)].index;
                const auto process_time_it = process_time_map.find(BuildProcessTimeKey(prev_job_id, prev_process_id, schedule_item.machine_id));
                graph[index2][index1] = process_time_it == process_time_map.end() ? 0 : process_time_it->second;
            }
        }
    }

    // 遍历待加工工件列表jobList,按照工件的工序，添加工序之间的关系
    for (const auto &job: jobList) {
        const auto job_it = job_name_map.find(job.substr(0, job.find('-')));
        if (job_it == job_name_map.end()) {
            continue;
        }
        const Job &jobItem = *job_it->second;
        const int job_id = jobItem.get_job_id();
        const int jobProcessCount = jobItem.get_process_count();
        for (int i = 0; i < jobProcessCount - 1; ++i) {
            const auto current_it = process_route_map.find(BuildProcessKey(job_id, i));
            const auto next_it = process_route_map.find(BuildProcessKey(job_id, i + 1));
            if (current_it == process_route_map.end() || next_it == process_route_map.end()) {
                continue;
            }
            const int machine_id = current_it->second.machine_id;
            const auto process_time_it = process_time_map.find(BuildProcessTimeKey(job_id, i, machine_id));
            graph[current_it->second.index][next_it->second.index] =
                process_time_it == process_time_map.end() ? 0 : process_time_it->second;
        }
    }
    if (flag) {
        schedule.set_graph(graph);
        schedule.set_processList(processList);
        CalculateTotalTime(schedule);
        CalculateTotalFailureRate(schedule, machines);
    }
    return graph;
}

int CalculateTotalFailureRate(Schedule &schedule, const std::vector<Machine> &machines){
    double total_failure_rate = 0.0;
    const std::vector<Schedule_item> schedule_items = schedule.get_schedule_items();
    for (int i = 0; i < schedule_items.size(); i++) {
        const int machine_id = schedule_items[i].machine_id;
        for (const auto &machine : machines) {
            if (machine.get_machine_id() == machine_id) {
                total_failure_rate += machine.get_failure_rate() * schedule_items[i].process_count;
                break;
            }
        }
    }
    schedule.set_total_failure_rate(total_failure_rate);
    return total_failure_rate;
}

int CalculateTotalTime(Schedule &schedule) {
    const std::vector<std::vector<int>> &graph = schedule.get_graph();
    const int processCount = graph.size();
    std::vector<int> inDegree(processCount, 0);  // 记录每个节点的入度
    for (int i = 0; i < processCount; ++i) {
        for (int j = 0; j < processCount; ++j) {
            if (graph[i][j] != -1) {
                inDegree[j]++;
            }
        }
    }
    // 获取拓扑排序的列表
    std::vector<int> startTime(processCount, 0);
    int visited_count = 0;
    std::queue<int> zeroInDegreeQueue;
    for (int i = 0; i < processCount; ++i) {
        if (inDegree[i] == 0) {
            zeroInDegreeQueue.push(i);
        }
    }
    while (!zeroInDegreeQueue.empty()) {
        const int temp = zeroInDegreeQueue.front();
        zeroInDegreeQueue.pop();
        ++visited_count;
        for (int i = 0; i < processCount; ++i) {
            if (graph[temp][i] != -1) {
                startTime[i] = std::max(startTime[i], startTime[temp] + graph[temp][i]);
                inDegree[i]--;
                if (inDegree[i] == 0) {
                    zeroInDegreeQueue.push(i);
                }
            }
        }
    }
    // std::cout << "Topological sort: ";
    // for (int i = 0; i<toPoSort.size(); i++) {
    //     std::cout << toPoSort[i] << " ";
    // }
    // std::cout << std::endl;

    // 检查是否存在环
    if (visited_count != processCount) {
        throw std::runtime_error("calculate time: The graph contains a cycle and cannot be topologically sorted.");
    }
    const int ans = processCount == 0 ? 0 : startTime[processCount - 1];
    schedule.set_start_time(startTime);
    schedule.set_TotalTime(ans);
    return ans;
}

// 设置停止标志，当总优化时间超过指定时间或搜索次数超过指定次数时，停止搜索
bool CheckStopFlag(const int &optimization_start_time,
                   const int &count,
                   const int repeat_count,
                   const int max_repeat_count,
                   const int total_time_limit_seconds) {
    const int now = time(nullptr);
    if (now - optimization_start_time >= total_time_limit_seconds) {
        return false;
    }
    if (count > 1200) {
        return false;
    }
    if (repeat_count>max_repeat_count) {
        return false;
    }
    return true;
}
