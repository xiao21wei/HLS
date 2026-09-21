//
// Created by 28898 on 25-4-14.
//

#ifndef TABUSEARCH_H
#define TABUSEARCH_H
#include <deque>
#include <string>
#include <unordered_set>
#include <vector>

#include "Job.h"
#include "Schedule.h"
#include "Machine.h"
#include "SearchDeadline.h"

// 设置禁忌搜索独立随机数生成器的种子。
void SetTabuSearchSeed(unsigned int random_seed);

// 基于关键块、多机器插入邻域和移动属性禁忌表的禁忌搜索。
Schedule TabuSearch(const Schedule &schedule, const std::vector<Job> &jobs, const std::vector<std::string> &jobList, int tabu_list_length, int max_iter_count, std::vector<Machine> &machines, SearchDeadline deadline = SearchDeadline::max(), bool use_persistent_state = true);

// 迁移邻域搜索：枚举关键工序的可选机器和可行插入位置。
std::vector<Schedule> MoveProcessSearch(const Schedule &schedule, const std::vector<Job> &jobs, const std::vector<std::string> &jobList, std::vector<Machine> &machines);

// 顺序邻域搜索：关键块 N5/N6 移动以及关键工序同机重插。
std::vector<Schedule> ExchangeProcessSearch(const Schedule &schedule, const std::vector<Job> &jobs, const std::vector<std::string> &jobList, std::vector<Machine> &machines);

// 获取调度方案的字符串表示
std::string ScheduleToString(const Schedule &schedule);

// 判断调度方案是否在禁忌列表中
bool IsInTabuList(const Schedule &schedule, const std::unordered_set<std::string> &tabu_set);

// 添加调度方案到禁忌列表
void AddScheduleToTabuList(const Schedule &schedule, std::deque<std::string> &tabu_queue, std::unordered_set<std::string> &tabu_set, int tabu_list_length);
#endif //TABUSEARCH_H
