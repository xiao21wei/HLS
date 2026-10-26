#include "AdaptiveSearch.h"

#include <array>
#include <stdexcept>

namespace {

void RecordFailure(
    AdaptiveSearchController &controller,
    const SearchDecision &decision
) {
    controller.Record(decision.method, 100, 100, 100, 100, 0.01);
}

void VerifyDisabledMethod(const SearchMethod disabled_method) {
    std::array<bool, 4> enabled{true, true, true, true};
    enabled[static_cast<int>(disabled_method)] = false;
    AdaptiveSearchController controller(
        0.60, 0.50, 1234U, true, false, enabled, true
    );
    for (int iteration = 0; iteration < 256; ++iteration) {
        if (iteration % 17 == 0) {
            controller.ScheduleIntensification();
        }
        const SearchDecision decision = controller.Select(20, 0.5);
        if (decision.method == disabled_method) {
            throw std::runtime_error("Disabled method was selected");
        }
        RecordFailure(controller, decision);
    }
}

}  // namespace

int main() {
    AdaptiveSearchController default_controller(0.60, 0.50, 99U, true, false);
    if (default_controller.Select(0, 1.0).method != SearchMethod::Greedy ||
        default_controller.Select(0, 1.0).method != SearchMethod::Tabu) {
        throw std::runtime_error("Default intensification order changed");
    }

    VerifyDisabledMethod(SearchMethod::Greedy);
    VerifyDisabledMethod(SearchMethod::Genetic);
    VerifyDisabledMethod(SearchMethod::Tabu);
    VerifyDisabledMethod(SearchMethod::Guided);

    std::array<bool, 4> genetic_only{false, true, false, false};
    AdaptiveSearchController fixed_controller(
        0.60, 0.50, 5678U, false, false, genetic_only, false
    );
    for (int iteration = 0; iteration < 32; ++iteration) {
        const SearchDecision decision = fixed_controller.Select(20, 0.5);
        if (decision.method != SearchMethod::Genetic) {
            throw std::runtime_error("Fixed selector ignored enabled methods");
        }
        RecordFailure(fixed_controller, decision);
    }
    return 0;
}
