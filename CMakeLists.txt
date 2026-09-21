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
        RandomSearch.cpp
        RandomSearch.h
        TabuSearch.cpp
        TabuSearch.h
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
