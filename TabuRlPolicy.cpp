#include "TabuRlPolicy.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char *kModelMagic = "HLS_GREEDY_RL_V1";
constexpr int kMaximumHiddenSize = 512;

struct TabuRlModel {
    int hidden_size = 0;
    std::vector<double> input_weights;
    std::vector<double> hidden_bias;
    std::vector<double> output_weights;
    double output_bias = 0.0;
};

TabuRlRuntimeConfig &MutableConfig() {
    static TabuRlRuntimeConfig config;
    return config;
}

TabuRlStats &MutableStats() {
    static TabuRlStats stats;
    return stats;
}

TabuRlModel &MutableModel() {
    static TabuRlModel model;
    return model;
}

std::mt19937 &PolicyRng() {
    static std::mt19937 random_engine(88675123U);
    return random_engine;
}

bool FlagEnabled(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "off" && text != "no";
}

bool EnvironmentFlag(const char *name, const bool fallback) {
    return std::getenv(name) == nullptr ? fallback : FlagEnabled(name);
}

int EnvironmentInteger(const char *name, const int fallback, const int minimum) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    try {
        return std::max(minimum, std::stoi(value));
    } catch (const std::exception &) {
        return fallback;
    }
}

double EnvironmentDouble(
    const char *name,
    const double fallback,
    const double minimum,
    const double maximum
) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    try {
        const double parsed = std::stod(value);
        return std::isfinite(parsed)
            ? std::clamp(parsed, minimum, maximum)
            : fallback;
    } catch (const std::exception &) {
        return fallback;
    }
}

bool ReadValues(
    std::istream &input,
    std::vector<double> &values,
    const std::size_t count
) {
    values.assign(count, 0.0);
    for (double &value : values) {
        if (!(input >> value) || !std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

bool LoadModel(
    const std::string &path,
    TabuRlModel &model,
    std::string &error
) {
    std::ifstream input(path);
    if (!input) {
        error = "model file not found: " + path;
        return false;
    }
    std::string magic;
    int input_size = 0;
    int hidden_size = 0;
    if (!(input >> magic >> input_size >> hidden_size) ||
        magic != kModelMagic || input_size != kGreedyRlFeatureCount ||
        hidden_size <= 0 || hidden_size > kMaximumHiddenSize) {
        error = "unsupported model shape or version";
        return false;
    }
    TabuRlModel loaded;
    loaded.hidden_size = hidden_size;
    if (!ReadValues(
            input,
            loaded.input_weights,
            static_cast<std::size_t>(hidden_size) * kGreedyRlFeatureCount
        ) ||
        !ReadValues(input, loaded.hidden_bias, hidden_size) ||
        !ReadValues(input, loaded.output_weights, hidden_size) ||
        !(input >> loaded.output_bias) ||
        !std::isfinite(loaded.output_bias)) {
        error = "model weights are incomplete or non-finite";
        return false;
    }
    model = std::move(loaded);
    return true;
}

double ScoreAction(
    const TabuRlModel &model,
    const GreedyRlFeatures &features
) {
    double score = model.output_bias;
    for (int hidden = 0; hidden < model.hidden_size; ++hidden) {
        double activation = model.hidden_bias[hidden];
        const std::size_t offset =
            static_cast<std::size_t>(hidden) * kGreedyRlFeatureCount;
        for (int feature = 0; feature < kGreedyRlFeatureCount; ++feature) {
            activation += model.input_weights[offset + feature] *
                          features[feature];
        }
        score += model.output_weights[hidden] * std::tanh(activation);
    }
    return score;
}

double ScoreRule(const GreedyRlFeatures &features) {
    return 10.0 * features[3] +
           8.0 * features[7] +
           0.50 * features[10] +
           0.25 * features[8] +
           0.10 * features[5] -
           1.00 * features[13];
}

int DynamicCandidateLimit(
    const TabuRlRuntimeConfig &config,
    const int operation_count
) {
    int limit = config.candidate_limit;
    if (!config.dynamic_candidate_limit) {
        return limit;
    }
    if (operation_count >= 200) {
        return std::max(limit, 768);
    }
    if (operation_count >= 160) {
        return std::max(limit, 512);
    }
    if (operation_count >= 80) {
        return std::max(limit, 384);
    }
    return limit;
}

}  // namespace

void ConfigureTabuRlPolicyFromEnvironment(const unsigned int random_seed) {
    TabuRlRuntimeConfig &config = MutableConfig();
    config = TabuRlRuntimeConfig{};
    MutableStats() = TabuRlStats{};
    MutableModel() = TabuRlModel{};
    PolicyRng().seed(random_seed);

    config.requested = FlagEnabled("HLS_TABU_RL");
    if (const char *path = std::getenv("HLS_TABU_RL_MODEL")) {
        config.model_path = path;
    } else if (const char *path = std::getenv("HLS_GREEDY_RL_MODEL")) {
        config.model_path = path;
    }
    config.candidate_limit = EnvironmentInteger(
        "HLS_TABU_RL_TOP_K",
        config.candidate_limit,
        1
    );
    config.dynamic_candidate_limit = EnvironmentFlag(
        "HLS_TABU_RL_DYNAMIC_TOP_K",
        config.dynamic_candidate_limit
    );
    config.fallback_on_miss = EnvironmentFlag(
        "HLS_TABU_RL_FALLBACK_ON_MISS",
        config.fallback_on_miss
    );
    config.fallback_on_non_improvement = EnvironmentFlag(
        "HLS_TABU_RL_FALLBACK_ON_NON_IMPROVEMENT",
        config.fallback_on_non_improvement
    );
    config.verify_full_neighborhood = EnvironmentFlag(
        "HLS_TABU_RL_VERIFY_FULL_NEIGHBORHOOD",
        config.verify_full_neighborhood
    );
    config.minimum_per_neighborhood = EnvironmentInteger(
        "HLS_TABU_RL_MIN_PER_NEIGHBORHOOD",
        config.minimum_per_neighborhood,
        0
    );
    config.temperature = EnvironmentDouble(
        "HLS_TABU_RL_TEMPERATURE",
        config.temperature,
        0.05,
        20.0
    );
    config.exploration_fraction = EnvironmentDouble(
        "HLS_TABU_RL_EXPLORATION",
        config.exploration_fraction,
        0.0,
        0.5
    );
    config.rule_fraction = EnvironmentDouble(
        "HLS_TABU_RL_RULE_FRACTION",
        config.rule_fraction,
        0.0,
        0.5
    );

    if (!config.requested) {
        config.status = "disabled by HLS_TABU_RL";
        return;
    }
    std::string error;
    if (!LoadModel(config.model_path, MutableModel(), error)) {
        config.status = error + "; using baseline Tabu";
        return;
    }
    config.enabled = true;
    config.status = "Greedy model loaded for Tabu candidates";
}

const TabuRlRuntimeConfig &GetTabuRlRuntimeConfig() {
    return MutableConfig();
}

const TabuRlStats &GetTabuRlStats() {
    return MutableStats();
}

bool TabuRlPolicyEnabled() {
    return MutableConfig().enabled;
}

void RecordTabuRlNeighborhoodSafeguards(const long long count) {
    MutableStats().neighborhood_safeguards += count;
}

void RecordTabuRlFallback() {
    ++MutableStats().fallback_calls;
}

void RecordTabuRlNonImprovementFallback() {
    ++MutableStats().non_improvement_fallback_calls;
}

void RecordTabuRlIntensificationExpansion() {
    ++MutableStats().intensification_expansion_calls;
}

void RecordTabuRlVerification() {
    ++MutableStats().verification_calls;
}

std::vector<std::size_t> SelectTabuRlCandidates(
    const std::vector<GreedyRlFeatures> &features,
    const int maximum_candidates,
    const int operation_count
) {
    if (features.empty() || maximum_candidates <= 0) {
        return {};
    }
    TabuRlStats &stats = MutableStats();
    ++stats.selection_calls;
    stats.generated_candidates += static_cast<long long>(features.size());
    const TabuRlRuntimeConfig &config = MutableConfig();
    const int selection_limit = std::min({
        static_cast<int>(features.size()),
        maximum_candidates,
        DynamicCandidateLimit(config, operation_count)
    });
    std::vector<std::size_t> indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (!config.enabled || static_cast<int>(features.size()) <= selection_limit) {
        indices.resize(selection_limit);
        stats.selected_candidates += selection_limit;
        if (config.enabled) {
            stats.model_candidates += selection_limit;
        }
        return indices;
    }

    std::vector<double> model_scores(features.size(), 0.0);
    std::vector<double> rule_scores(features.size(), 0.0);
    for (std::size_t index = 0; index < features.size(); ++index) {
        model_scores[index] = ScoreAction(MutableModel(), features[index]);
        rule_scores[index] = ScoreRule(features[index]);
    }
    std::stable_sort(
        indices.begin(),
        indices.end(),
        [&model_scores](const std::size_t left, const std::size_t right) {
            if (model_scores[left] != model_scores[right]) {
                return model_scores[left] > model_scores[right];
            }
            return left < right;
        }
    );
    const int exploration_count = std::min(selection_limit, static_cast<int>(
        std::lround(selection_limit * config.exploration_fraction)
    ));
    const int rule_count = std::min(
        selection_limit - exploration_count,
        static_cast<int>(std::lround(selection_limit * config.rule_fraction))
    );
    const int model_count = selection_limit - exploration_count - rule_count;

    std::vector<std::size_t> selected;
    selected.reserve(selection_limit);
    std::vector<bool> used(features.size(), false);
    auto add = [&](const std::size_t index) {
        if (used[index]) {
            return false;
        }
        used[index] = true;
        selected.push_back(index);
        return true;
    };
    for (const std::size_t index : indices) {
        if (static_cast<int>(selected.size()) >= model_count) {
            break;
        }
        if (add(index)) {
            ++stats.model_candidates;
        }
    }

    const int prefix_rule_count = (rule_count + 1) / 2;
    const int after_prefix_target = model_count + prefix_rule_count;
    for (std::size_t index = 0;
         index < features.size() &&
         static_cast<int>(selected.size()) < after_prefix_target;
         ++index) {
        if (add(index)) {
            ++stats.rule_candidates;
        }
    }

    std::vector<std::size_t> rule_indices(features.size());
    std::iota(rule_indices.begin(), rule_indices.end(), 0);
    std::stable_sort(
        rule_indices.begin(),
        rule_indices.end(),
        [&rule_scores](const std::size_t left, const std::size_t right) {
            if (rule_scores[left] != rule_scores[right]) {
                return rule_scores[left] > rule_scores[right];
            }
            return left < right;
        }
    );
    for (const std::size_t index : rule_indices) {
        if (static_cast<int>(selected.size()) >= model_count + rule_count) {
            break;
        }
        if (add(index)) {
            ++stats.rule_candidates;
        }
    }

    std::vector<std::size_t> remaining;
    for (std::size_t index = 0; index < features.size(); ++index) {
        if (!used[index]) {
            remaining.push_back(index);
        }
    }
    while (static_cast<int>(selected.size()) < selection_limit &&
           !remaining.empty()) {
        double maximum_score = -std::numeric_limits<double>::infinity();
        for (const std::size_t index : remaining) {
            maximum_score = std::max(maximum_score, model_scores[index]);
        }
        std::vector<double> weights;
        weights.reserve(remaining.size());
        for (const std::size_t index : remaining) {
            weights.push_back(std::exp(std::clamp(
                (model_scores[index] - maximum_score) / config.temperature,
                -50.0,
                0.0
            )));
        }
        const int chosen = std::discrete_distribution<int>(
            weights.begin(),
            weights.end()
        )(PolicyRng());
        add(remaining[chosen]);
        ++stats.exploration_candidates;
        remaining.erase(remaining.begin() + chosen);
    }
    stats.selected_candidates += static_cast<long long>(selected.size());
    return selected;
}
