#ifndef ADAPTIVESEARCH_H
#define ADAPTIVESEARCH_H

#include <array>
#include <deque>
#include <random>

enum class SearchMethod {
    Greedy = 0,
    Genetic = 1,
    Tabu = 2
};

struct SearchDecision {
    SearchMethod method = SearchMethod::Greedy;
    bool intensification = false;
    bool diversified_start = false;
};

struct SearchMethodStats {
    int calls = 0;
    int improvements = 0;
    int global_improvements = 0;
    int consecutive_failures = 0;
    double total_relative_gain = 0.0;
    double total_global_relative_gain = 0.0;
    double total_seconds = 0.0;
    double recent_reward = 0.0;
    int last_called_at = -1;
};

class AdaptiveSearchController {
public:
    AdaptiveSearchController(
        double search_mode_preference,
        double genetic_preference,
        unsigned int random_seed
    );

    SearchDecision Select(int stagnation_count, double remaining_fraction);
    void Record(
        SearchMethod method,
        int local_before_makespan,
        int local_after_makespan,
        int global_before_makespan,
        int global_after_makespan,
        double elapsed_seconds
    );
    void ScheduleIntensification();
    int SelectArchiveIndex(int archive_size);
    const SearchMethodStats &GetStats(SearchMethod method) const;

private:
    static int MethodIndex(SearchMethod method);
    double Score(
        SearchMethod method,
        int stagnation_count,
        double remaining_fraction
    ) const;
    SearchMethod SelectExplorationMethod(
        const std::array<bool, 3> &eligible_methods
    );

    std::array<double, 3> initial_preferences_{};
    std::array<SearchMethodStats, 3> statistics_{};
    std::deque<SearchDecision> pending_intensification_;
    std::mt19937 random_engine_;
    int total_calls_ = 0;
};

const char *SearchMethodName(SearchMethod method);

#endif  // ADAPTIVESEARCH_H
