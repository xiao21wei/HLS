//
// Created by luopw on 24-12-1.
//

#ifndef GREEDYSEARCH_H
#define GREEDYSEARCH_H
#include <vector>

#include "Job.h"
#include "Schedule.h"
#include "Machine.h"
#include "SearchDeadline.h"

// 基于关键工序的迭代 best-improvement 搜索
Schedule GreedySearch(const Schedule &schedule, const std::vector<Job> &jobs, const std::vector<std::string> &jobList, int iter_count, std::vector<Machine> &machines, SearchDeadline deadline = SearchDeadline::max());

// 贪心调整
Schedule GreedyAdjust(const Schedule &schedule, const std::vector<Job> &jobs, const std::vector<std::string> &jobList, std::vector<Machine> &machines);

// 判断是否存在环
bool hasCycle(const std::vector<std::vector<int>> &adjMatrix);

// 在同一机器上重新插入关键工序，选择 makespan 最小的改进方案
Schedule ExchangeNeighborSearch(const Schedule &schedule, const std::vector<Job> &jobs, const std::vector<std::string> &jobList, std::vector<std::string> &processlist, bool &flag, std::vector<Machine> &machines);

// 按机器负载、加工时间和开始时间对关键工序排序
std::vector<std::string> ScoreProcessForExchange(const Schedule &schedule, const std::vector<Job> &jobs, const std::vector<std::string> &jobList);

// 获取工序在调度方案中的索引
int GetItemIndex(const Schedule &schedule, const int job_id, const int process_id);

// 根据工序，在调度方案中获取到工序所在的机器编号
void GetMachineIdAndItemIdByProcess(const std::vector<Schedule_item> &schedule_items, const std::string &process, int type, int &x, int &y);

// 枚举关键工序的可选机器和插入位置，选择 makespan 最小的改进方案
Schedule MoveNeighborSearch(const Schedule &schedule, const std::vector<Job> &jobs, const std::vector<std::string> &jobList, std::vector<std::string> &processlist, bool &flag, std::vector<Machine> &machines);

// 根据工序的编号，获取工序的开始时间
int GetStartTimeByProcess(const Schedule &schedule, const int job_id, const int process_id);

// 按机器加工时间改进潜力和机器负载对柔性关键工序排序
std::vector<std::string> ScoreProcessForMove(const Schedule &schedule, const std::vector<Job> &jobs, const std::vector<std::string> &jobList);

// 计算可调度工序的各个机器的工序数量，返回最大差值
int GetDifferFromIdealMachine(const Schedule &schedule, const std::vector<Job> &jobs, const int job_id, const int process_id);

// 获取机器上的工序数量
int GetProcessCountByMachineId(const Schedule &schedule, const int machine_id);

// 根据工件名称查找工件
Job SelectJobByJobId(const std::vector<Job> &jobs, const int job_id);
#endif //GREEDYSEARCH_H
