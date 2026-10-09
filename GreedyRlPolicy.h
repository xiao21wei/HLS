#ifndef GREEDY_RL_POLICY_H
#define GREEDY_RL_POLICY_H

#include <array>
#include <cstddef>
#include <string>
#include <vector>

constexpr int kGreedyRlFeatureCount = 16;
using GreedyRlFeatures = std::array<double, kGreedyRlFeatureCount>;

struct GreedyRlRuntimeConfig {
    bool requested = false;
    bool enabled = false;
    std::string model_path = "rl_greedy/models/greedy_policy.txt";
    int candidate_limit = 96;
    bool dynamic_candidate_limit = true;
    bool fallback_on_miss = true;
    bool verify_with_baseline = false;
    double temperature = 1.0;
    double exploration_fraction = 0.15;
    double rule_fraction = 0.25;
    std::string status = "disabled";
};

struct GreedyRlStats {
    long long selection_calls = 0;
    long long generated_candidates = 0;
    long long selected_candidates = 0;
    long long model_candidates = 0;
    long long rule_candidates = 0;
    long long exploration_candidates = 0;
    long long fallback_calls = 0;
    long long verification_calls = 0;
};

void ConfigureGreedyRlPolicyFromEnvironment(unsigned int random_seed);

const GreedyRlRuntimeConfig &GetGreedyRlRuntimeConfig();

const GreedyRlStats &GetGreedyRlStats();

bool GreedyRlPolicyEnabled();

void RecordGreedyRlFallback();

void RecordGreedyRlVerification();

std::vector<std::size_t> SelectGreedyRlCandidates(
    const std::vector<GreedyRlFeatures> &features,
    int maximum_candidates,
    int operation_count
);

#endif  // GREEDY_RL_POLICY_H
