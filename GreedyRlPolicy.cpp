#include "GreedyRlPolicy.h"

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

struct GreedyRlModel {
    int hidden_size = 0;
    std::vector<double> input_weights;
    std::vector<double> hidden_bias;
    std::vector<double> output_weights;
    double output_bias = 0.0;
};

GreedyRlRuntimeConfig &MutableConfig() {
    static GreedyRlRuntimeConfig config;
    return config;
}

GreedyRlStats &MutableStats() {
    static GreedyRlStats stats;
    return stats;
}

GreedyRlModel &MutableModel() {
    static GreedyRlModel model;
    return model;
}

std::mt19937 &PolicyRng() {
    static std::mt19937 random_engine(521288629U);
    return random_engine;
}

bool EnvironmentFlagEnabled(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "off" && text != "no";
}

bool EnvironmentFlag(
    const char *name,
    const bool fallback
) {
    return std::getenv(name) == nullptr
        ? fallback
        : EnvironmentFlagEnabled(name);
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
        if (!std::isfinite(parsed)) {
            return fallback;
        }
        return std::clamp(parsed, minimum, maximum);
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
    GreedyRlModel &model,
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
    if (!(input >> magic >> input_size >> hidden_size)) {
        error = "invalid model header";
        return false;
    }
    if (magic != kModelMagic || input_size != kGreedyRlFeatureCount ||
        hidden_size <= 0 || hidden_size > kMaximumHiddenSize) {
        error = "unsupported model shape or version";
        return false;
    }

    GreedyRlModel loaded;
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
    const GreedyRlModel &model,
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
    const GreedyRlRuntimeConfig &config,
    const int operation_count
) {
    int limit = config.candidate_limit;
    if (!config.dynamic_candidate_limit) {
        return limit;
    }
    if (operation_count >= 200) {
        return std::max(limit, 384);
    }
    if (operation_count >= 160) {
        return std::max(limit, 256);
    }
    if (operation_count >= 80) {
        return std::max(limit, 192);
    }
    return limit;
}

}  // namespace

void ConfigureGreedyRlPolicyFromEnvironment(const unsigned int random_seed) {
    GreedyRlRuntimeConfig &config = MutableConfig();
    config = GreedyRlRuntimeConfig{};
    MutableStats() = GreedyRlStats{};
    MutableModel() = GreedyRlModel{};
    PolicyRng().seed(random_seed);

    config.requested = EnvironmentFlagEnabled("HLS_GREEDY_RL");
    if (const char *model_path = std::getenv("HLS_GREEDY_RL_MODEL")) {
        config.model_path = model_path;
    }
    config.candidate_limit = EnvironmentInteger(
        "HLS_GREEDY_RL_TOP_K",
        config.candidate_limit,
        1
    );
    config.dynamic_candidate_limit = EnvironmentFlag(
        "HLS_GREEDY_RL_DYNAMIC_TOP_K",
        config.dynamic_candidate_limit
    );
    config.fallback_on_miss = EnvironmentFlag(
        "HLS_GREEDY_RL_FALLBACK_ON_MISS",
        config.fallback_on_miss
    );
    config.verify_with_baseline = EnvironmentFlag(
        "HLS_GREEDY_RL_VERIFY_WITH_BASELINE",
        config.verify_with_baseline
    );
    config.temperature = EnvironmentDouble(
        "HLS_GREEDY_RL_TEMPERATURE",
        config.temperature,
        0.05,
        20.0
    );
    config.exploration_fraction = EnvironmentDouble(
        "HLS_GREEDY_RL_EXPLORATION",
        config.exploration_fraction,
        0.0,
        0.5
    );
    config.rule_fraction = EnvironmentDouble(
        "HLS_GREEDY_RL_RULE_FRACTION",
        config.rule_fraction,
        0.0,
        0.5
    );

    if (!config.requested) {
        config.status = "disabled by HLS_GREEDY_RL";
        return;
    }

    std::string error;
    if (!LoadModel(config.model_path, MutableModel(), error)) {
        config.status = error + "; using baseline Greedy";
        return;
    }
    config.enabled = true;
    config.status = "model loaded";
}

const GreedyRlRuntimeConfig &GetGreedyRlRuntimeConfig() {
    return MutableConfig();
}

const GreedyRlStats &GetGreedyRlStats() {
    return MutableStats();
}

bool GreedyRlPolicyEnabled() {
    return MutableConfig().enabled;
}

void SetGreedyRlSamplingSeed(const unsigned int random_seed) {
    PolicyRng().seed(random_seed);
}

void RecordGreedyRlFallback() {
    ++MutableStats().fallback_calls;
}

void RecordGreedyRlVerification() {
    ++MutableStats().verification_calls;
}

std::vector<std::size_t> SelectGreedyRlCandidates(
    const std::vector<GreedyRlFeatures> &features,
    const int maximum_candidates,
    const int operation_count
) {
    std::vector<std::size_t> indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (features.empty() || maximum_candidates <= 0) {
        return {};
    }

    GreedyRlStats &stats = MutableStats();
    ++stats.selection_calls;
    stats.generated_candidates += static_cast<long long>(features.size());

    const GreedyRlRuntimeConfig &config = MutableConfig();
    const int selection_limit = std::min({
        static_cast<int>(features.size()),
        maximum_candidates,
        DynamicCandidateLimit(config, operation_count)
    });
    if (!config.enabled || static_cast<int>(features.size()) <= selection_limit) {
        indices.resize(selection_limit);
        stats.selected_candidates += selection_limit;
        if (config.enabled) {
            stats.model_candidates += selection_limit;
        }
        return indices;
    }

    const GreedyRlModel &model = MutableModel();
    std::vector<double> scores(features.size(), 0.0);
    std::vector<double> rule_scores(features.size(), 0.0);
    for (std::size_t index = 0; index < features.size(); ++index) {
        scores[index] = ScoreAction(model, features[index]);
        rule_scores[index] = ScoreRule(features[index]);
    }
    std::stable_sort(
        indices.begin(),
        indices.end(),
        [&scores](const std::size_t left, const std::size_t right) {
            if (scores[left] != scores[right]) {
                return scores[left] > scores[right];
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
    std::vector<bool> was_selected(features.size(), false);
    auto add_candidate = [&](const std::size_t index) {
        if (was_selected[index]) {
            return false;
        }
        was_selected[index] = true;
        selected.push_back(index);
        return true;
    };
    for (const std::size_t index : indices) {
        if (static_cast<int>(selected.size()) >= model_count) {
            break;
        }
        if (add_candidate(index)) {
            ++stats.model_candidates;
        }
    }

    const int prefix_rule_count = (rule_count + 1) / 2;
    const int after_prefix_target = model_count + prefix_rule_count;
    for (std::size_t index = 0;
         index < features.size() &&
         static_cast<int>(selected.size()) < after_prefix_target;
         ++index) {
        if (add_candidate(index)) {
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
    const int after_rule_target = model_count + rule_count;
    for (const std::size_t index : rule_indices) {
        if (static_cast<int>(selected.size()) >= after_rule_target) {
            break;
        }
        if (add_candidate(index)) {
            ++stats.rule_candidates;
        }
    }

    std::vector<std::size_t> remaining;
    remaining.reserve(features.size() - selected.size());
    for (std::size_t index = 0; index < features.size(); ++index) {
        if (!was_selected[index]) {
            remaining.push_back(index);
        }
    }

    for (int sample = 0;
         static_cast<int>(selected.size()) < selection_limit &&
         sample < exploration_count && !remaining.empty();
         ++sample) {
        double maximum_score = -std::numeric_limits<double>::infinity();
        for (const std::size_t index : remaining) {
            maximum_score = std::max(maximum_score, scores[index]);
        }
        std::vector<double> weights;
        weights.reserve(remaining.size());
        for (const std::size_t index : remaining) {
            weights.push_back(std::exp(std::clamp(
                (scores[index] - maximum_score) / config.temperature,
                -50.0,
                0.0
            )));
        }
        const int chosen = std::discrete_distribution<int>(
            weights.begin(),
            weights.end()
        )(PolicyRng());
        add_candidate(remaining[chosen]);
        ++stats.exploration_candidates;
        remaining.erase(remaining.begin() + chosen);
    }

    for (const std::size_t index : remaining) {
        if (static_cast<int>(selected.size()) >= selection_limit) {
            break;
        }
        if (add_candidate(index)) {
            ++stats.exploration_candidates;
        }
    }

    stats.selected_candidates += static_cast<long long>(selected.size());
    return selected;
}
