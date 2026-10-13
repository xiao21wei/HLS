#ifndef TABU_RL_POLICY_H
#define TABU_RL_POLICY_H

#include <cstddef>
#include <string>
#include <vector>

#include "GreedyRlPolicy.h"

struct TabuRlRuntimeConfig {
    bool requested = false;
    bool enabled = false;
    std::string model_path = "rl_greedy/models/greedy_policy.txt";
    int candidate_limit = 256;
    bool dynamic_candidate_limit = true;
    bool fallback_on_miss = true;
    bool fallback_on_non_improvement = false;
    bool verify_full_neighborhood = false;
    int minimum_per_neighborhood = 16;
    double temperature = 1.0;
    double exploration_fraction = 0.20;
    double rule_fraction = 0.30;
    std::string status = "disabled";
};

struct TabuRlStats {
    long long selection_calls = 0;
    long long generated_candidates = 0;
    long long selected_candidates = 0;
    long long model_candidates = 0;
    long long rule_candidates = 0;
    long long exploration_candidates = 0;
    long long neighborhood_safeguards = 0;
    long long fallback_calls = 0;
    long long non_improvement_fallback_calls = 0;
    long long intensification_expansion_calls = 0;
    long long verification_calls = 0;
};

void ConfigureTabuRlPolicyFromEnvironment(unsigned int random_seed);
void SetTabuRlSamplingSeed(unsigned int random_seed);

const TabuRlRuntimeConfig &GetTabuRlRuntimeConfig();

const TabuRlStats &GetTabuRlStats();

bool TabuRlPolicyEnabled();

void RecordTabuRlNeighborhoodSafeguards(long long count);

void RecordTabuRlFallback();

void RecordTabuRlNonImprovementFallback();

void RecordTabuRlIntensificationExpansion();

void RecordTabuRlVerification();

std::vector<std::size_t> SelectTabuRlCandidates(
    const std::vector<GreedyRlFeatures> &features,
    int maximum_candidates,
    int operation_count
);

#endif  // TABU_RL_POLICY_H
