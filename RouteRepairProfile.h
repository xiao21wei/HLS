#ifndef ROUTE_REPAIR_PROFILE_H
#define ROUTE_REPAIR_PROFILE_H

#include <vector>

#include "Job.h"

bool IsLongRouteFlexibleProblem(const std::vector<Job> &jobs);

// Route-coupled repair is limited to structures where a 32-operation release
// can cover several useful neighboring route fragments.
bool IsRouteCoupledRepairProblem(const std::vector<Job> &jobs);

#endif  // ROUTE_REPAIR_PROFILE_H
