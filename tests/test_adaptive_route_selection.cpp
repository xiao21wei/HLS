#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "AdaptiveSearch.h"

int main() {
    unsetenv("HLS_GREEDY_SCORE_BONUS");
    unsetenv("HLS_GUIDED_SCORE_BONUS");
    const AdaptiveSearchController baseline(0.6, 0.5, 17U, true, false);
    const AdaptiveSearchController route_aware(0.6, 0.5, 17U, true, true);
    if (std::abs(route_aware.GetScoreBonus(SearchMethod::Greedy) -
                 baseline.GetScoreBonus(SearchMethod::Greedy) - 0.05) > 1e-12) {
        throw std::runtime_error("Route-aware Greedy bonus is incorrect");
    }
    if (std::abs(route_aware.GetScoreBonus(SearchMethod::Guided) -
                 baseline.GetScoreBonus(SearchMethod::Guided) + 0.05) > 1e-12) {
        throw std::runtime_error("Route-aware Guided penalty is incorrect");
    }
}
