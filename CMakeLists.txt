cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
project(production_scheduling)

set(CMAKE_CXX_STANDARD 20)

set(COPT_HOME "$ENV{COPT_HOME}" CACHE PATH "COPT installation directory")
if(NOT COPT_HOME)
    set(COPT_HOME "/home/luopw24/copt80")
endif()

add_executable(production_scheduling main.cpp
        AdaptiveSearch.cpp
        AdaptiveSearch.h
        SearchDeadline.h
        RouteRepairProfile.cpp
        RouteRepairProfile.h
        Order.cpp
        Order.h
        Job.cpp
        Job.h
        Machine.cpp
        Machine.h
        Schedule.cpp
        Schedule.h
        DataProc.cpp
        DataProc.h
        GreedySearch.cpp
        GreedySearch.h
        GreedyRlPolicy.cpp
        GreedyRlPolicy.h
        GuidedSearch.cpp
        GuidedSearch.h
        RandomSearch.cpp
        RandomSearch.h
        TabuSearch.cpp
        TabuSearch.h
        TabuRlPolicy.cpp
        TabuRlPolicy.h
        Experiment.cpp
        Experiment.h
        CoptInitialSolution.cpp
        CoptInitialSolution.h)

target_include_directories(production_scheduling PRIVATE
        "${COPT_HOME}/include"
        "${COPT_HOME}/include/coptcpp_inc")
target_link_directories(production_scheduling PRIVATE "${COPT_HOME}/lib")
target_link_libraries(production_scheduling PRIVATE copt_cpp)
set_target_properties(production_scheduling PROPERTIES
        BUILD_RPATH "${COPT_HOME}/lib"
        INSTALL_RPATH "${COPT_HOME}/lib")

include(CTest)
if(BUILD_TESTING)
    add_executable(test_route_repair_profile
            tests/test_route_repair_profile.cpp
            RouteRepairProfile.cpp
            Job.cpp)
    target_include_directories(test_route_repair_profile PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME route_repair_profile COMMAND test_route_repair_profile)

    add_executable(test_adaptive_route_selection
            tests/test_adaptive_route_selection.cpp
            AdaptiveSearch.cpp)
    target_include_directories(test_adaptive_route_selection PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME adaptive_route_selection COMMAND test_adaptive_route_selection)

    add_executable(test_adaptive_method_ablation
            tests/test_adaptive_method_ablation.cpp
            AdaptiveSearch.cpp)
    target_include_directories(test_adaptive_method_ablation PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME adaptive_method_ablation COMMAND test_adaptive_method_ablation)
endif()
