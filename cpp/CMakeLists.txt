cmake_minimum_required(VERSION 3.10)
project(FinanceManager)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create the pybind11 module
pybind11_add_module(finance finance.cpp)

# Set optimization flags for Release build
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Install the library
install(TARGETS finance DESTINATION .)
