#include "AdaptiveSearch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr int kMethodCount = 3;
constexpr int kMinimumInitialCalls = 2;
constexpr int kMildStagnation = 5;
constexpr int kDeepStagnation = 12;

double ClampProbability(const double value) {
    return std::clamp(value, 0.0, 1.0);
}

}  // namespace

AdaptiveSearchController::AdaptiveSearchController(
    const double search_mode_preference,
    const double genetic_preference,
    const unsigned int random_seed
) : random_engine_(random_seed) {
    const double search_preference = ClampProbability(search_mode_preference);
    const double genetic_share = ClampProbability(genetic_preference);
    initial_preferences_[MethodIndex(SearchMethod::Greedy)] =
        std::max(0.05, 1.0 - search_preference);
    initial_preferences_[MethodIndex(SearchMethod::Genetic)] =
        std::max(0.05, search_preference * genetic_share);
    initial_preferences_[MethodIndex(SearchMethod::Tabu)] =
        std::max(0.05, search_preference * (1.0 - genetic_share));

    const double preference_sum =
        initial_preferences_[0] +
        initial_preferences_[1] +
        initial_preferences_[2];
    for (double &preference : initial_preferences_) {
        preference /= preference_sum;
    }

    pending_intensification_.push_back({SearchMethod::Greedy, true, false});
    pending_intensification_.push_back({SearchMethod::Tabu, true, false});
}

SearchDecision AdaptiveSearchController::Select(
    const int stagnation_count,
    const double remaining_fraction
) {
    if (!pending_intensification_.empty()) {
        const SearchDecision decision = pending_intensification_.front();
        pending_intensification_.pop_front();
        return decision;
    }

    std::array<bool, kMethodCount> eligible_methods{true, true, true};
    bool has_method_without_long_failure = false;
    for (const auto &stats : statistics_) {
        has_method_without_long_failure =
            has_method_without_long_failure || stats.consecutive_failures < 6;
    }
    if (has_method_without_long_failure) {
        for (int index = 0; index < kMethodCount; ++index) {
            const SearchMethodStats &stats = statistics_[index];
            const bool periodic_probe =
                (total_calls_ + index * 4) % 12 == 0;
            if (stats.consecutive_failures >= 6 && !periodic_probe) {
                eligible_methods[index] = false;
            }
        }
    }

    SearchMethod initial_method = SearchMethod::Greedy;
    int smallest_call_count = std::numeric_limits<int>::max();
    double largest_preference = -1.0;
    bool needs_initial_sampling = false;
    for (int index = 0; index < kMethodCount; ++index) {
        if (!eligible_methods[index] ||
            statistics_[index].calls >= kMinimumInitialCalls) {
            continue;
        }
        if (statistics_[index].calls < smallest_call_count ||
            (statistics_[index].calls == smallest_call_count &&
             initial_preferences_[index] > largest_preference)) {
            smallest_call_count = statistics_[index].calls;
            largest_preference = initial_preferences_[index];
            initial_method = static_cast<SearchMethod>(index);
            needs_initial_sampling = true;
        }
    }
    if (needs_initial_sampling) {
        return {initial_method, false, false};
    }

    const double exploration_probability =
        (stagnation_count >= kDeepStagnation ? 0.20 : 0.08) +
        0.08 * ClampProbability(remaining_fraction);
    if (std::generate_canonical<double, 53>(random_engine_) <
        exploration_probability) {
        const SearchMethod method = SelectExplorationMethod(eligible_methods);
        return {
            method,
            false,
            method == SearchMethod::Genetic &&
                stagnation_count >= kDeepStagnation
        };
    }

    SearchMethod best_method = SearchMethod::Greedy;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int index = 0; index < kMethodCount; ++index) {
        if (!eligible_methods[index]) {
            continue;
        }
        const SearchMethod method = static_cast<SearchMethod>(index);
        const double method_score = Score(
            method,
            stagnation_count,
            remaining_fraction
        );
        if (method_score > best_score) {
            best_score = method_score;
            best_method = method;
        }
    }
    return {
        best_method,
        false,
        best_method == SearchMethod::Genetic &&
            stagnation_count >= kDeepStagnation
    };
}

void AdaptiveSearchController::Record(
    const SearchMethod method,
    const int local_before_makespan,
    const int local_after_makespan,
    const int global_before_makespan,
    const int global_after_makespan,
    const double elapsed_seconds
) {
    SearchMethodStats &stats = statistics_[MethodIndex(method)];
    ++stats.calls;
    ++total_calls_;

    double local_relative_gain = 0.0;
    if (local_before_makespan > 0 && local_after_makespan < local_before_makespan) {
        local_relative_gain = static_cast<double>(
            local_before_makespan - local_after_makespan
        ) / local_before_makespan;
        ++stats.improvements;
        stats.total_relative_gain += local_relative_gain;
    }
    double global_relative_gain = 0.0;
    if (global_before_makespan > 0 &&
        global_after_makespan < global_before_makespan) {
        global_relative_gain = static_cast<double>(
            global_before_makespan - global_after_makespan
        ) / global_before_makespan;
        ++stats.global_improvements;
        stats.total_global_relative_gain += global_relative_gain;
        stats.consecutive_failures = 0;
    } else {
        ++stats.consecutive_failures;
    }
    // A local improvement from an old archive member is useful as diversity,
    // but it must not outweigh a method that improves the global incumbent.
    const double credited_gain = global_relative_gain +
                                 0.05 * local_relative_gain;
    const double reward = std::min(
        1.0,
        credited_gain / std::max(0.05, elapsed_seconds)
    );
    stats.recent_reward = stats.calls == 1
        ? reward
        : 0.70 * stats.recent_reward + 0.30 * reward;
    stats.total_seconds += std::max(0.0, elapsed_seconds);
    stats.last_called_at = total_calls_;

}

void AdaptiveSearchController::ScheduleIntensification() {
    if (!pending_intensification_.empty()) {
        return;
    }
    pending_intensification_.push_back({SearchMethod::Greedy, true, false});
    pending_intensification_.push_back({SearchMethod::Tabu, true, false});
}

int AdaptiveSearchController::SelectArchiveIndex(const int archive_size) {
    if (archive_size <= 1) {
        return 0;
    }
    if (std::bernoulli_distribution(0.65)(random_engine_)) {
        return archive_size - 1;
    }
    return std::uniform_int_distribution<int>(0, archive_size - 2)(random_engine_);
}

const SearchMethodStats &AdaptiveSearchController::GetStats(
    const SearchMethod method
) const {
    return statistics_[MethodIndex(method)];
}

int AdaptiveSearchController::MethodIndex(const SearchMethod method) {
    return static_cast<int>(method);
}

double AdaptiveSearchController::Score(
    const SearchMethod method,
    const int stagnation_count,
    const double remaining_fraction
) const {
    const int index = MethodIndex(method);
    const SearchMethodStats &stats = statistics_[index];
    const double local_success_rate = stats.calls == 0
        ? 0.0
        : static_cast<double>(stats.improvements) / stats.calls;
    const double global_success_rate = stats.calls == 0
        ? 0.0
        : static_cast<double>(stats.global_improvements) / stats.calls;
    const double exploration_bonus = 0.18 * std::sqrt(
        std::log(static_cast<double>(total_calls_) + 2.0) /
        (static_cast<double>(stats.calls) + 1.0)
    );
    const double time_efficiency = std::min(
        1.0,
        20.0 * stats.total_global_relative_gain /
            std::max(0.1, stats.total_seconds)
    );
    const double failure_penalty = std::min(
        0.20,
        0.02 * stats.consecutive_failures
    );
    const double recency_bonus = stats.last_called_at < 0
        ? 0.10
        : std::min(0.12, 0.01 * (total_calls_ - stats.last_called_at));
    double score = 0.40 * stats.recent_reward +
                   0.25 * time_efficiency +
                   0.15 * global_success_rate +
                   0.05 * local_success_rate +
                   0.05 * initial_preferences_[index] +
                   exploration_bonus -
                   failure_penalty + recency_bonus;

    if (method == SearchMethod::Tabu && stagnation_count >= kMildStagnation) {
        score += std::min(0.10, stagnation_count * 0.005);
    }
    if (method == SearchMethod::Genetic && stagnation_count >= kDeepStagnation) {
        score += 0.12;
    }
    if (remaining_fraction <= 0.10 && method == SearchMethod::Greedy) {
        score += 0.08;
    }
    return score;
}

SearchMethod AdaptiveSearchController::SelectExplorationMethod(
    const std::array<bool, 3> &eligible_methods
) {
    std::vector<SearchMethod> methods;
    std::vector<double> weights;
    for (int index = 0; index < kMethodCount; ++index) {
        if (eligible_methods[index]) {
            methods.push_back(static_cast<SearchMethod>(index));
            weights.push_back(initial_preferences_[index]);
        }
    }
    std::discrete_distribution<int> distribution(weights.begin(), weights.end());
    return methods[distribution(random_engine_)];
}

const char *SearchMethodName(const SearchMethod method) {
    switch (method) {
        case SearchMethod::Greedy:
            return "Greedy Search";
        case SearchMethod::Genetic:
            return "Genetic Algorithm";
        case SearchMethod::Tabu:
            return "Tabu Search";
    }
    return "Unknown Search";
}
