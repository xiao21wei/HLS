//
// Created by luopw on 24-12-1.
//

#ifndef DATAPROC_H
#define DATAPROC_H
#include <vector>

#include "Job.h"
#include "Machine.h"
#include "Order.h"
#include "Schedule.h"


// 从文件中读取数据，初始化机器、工件、订单
void Init(std::vector<Machine> &machines, std::vector<Job> &jobs, std::vector<Order> &orders, std::string file_path);

// 将订单转换为待加工的工件列表
void OrderToJobList(const std::vector<Order> &orders, std::vector<std::string> &jobList);

Schedule GenerateSolution(std::string file_path, const std::vector<std::string> &jobList, const std::vector<Job> &jobs, std::vector<Machine> &machines);
// 生成初始解，使用考虑工件就绪时间和机器加工时间的最早完工策略
Schedule GenerateInitialSolution(const std::vector<std::string> &jobList, const std::vector<Job> &jobs, std::vector<Machine> &machines);

// 根据工件名称查找工件
Job SelectJobByJobName(const std::vector<Job> &jobs, const std::string &jobName);

// 根据机器id, 工件id, 工序id查询时间开销
int GetProcessTime(const std::vector<Job> &jobs, const int machine_id, const int job_id, const int process_id);

// 将调度方案转换为邻接矩阵
std::vector<std::vector<int>> ScheduleItemsToGraph(Schedule &schedule, std::vector<Schedule_item> &schedule_items, const std::vector<Job> &jobs, const std::vector<std::string> &jobList, bool flag, std::vector<Machine> &machines);

// 计算调度方案的总时间开销
int CalculateTotalTime(Schedule &schedule);

// 计算调度方案的总故障率
int CalculateTotalFailureRate(Schedule &schedule, const std::vector<Machine> &machines);

// 设置停止标志，当总优化时间超过指定时间或搜索次数超过指定次数时，停止搜索
bool CheckStopFlag(const int &optimization_start_time,
                   const int &count,
                   int repeat_count,
                   int max_repeat_count,
                   int total_time_limit_seconds);

// std::vector<std::vector<std::string>> ScheduleToProcessList(const Schedule &schedule);
#endif //DATAPROC_H
