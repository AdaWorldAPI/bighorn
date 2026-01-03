cmake_minimum_required(VERSION 3.14)
project(agi_thinking VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(AGI_THINKING_BUILD_TESTS "Build unit tests" OFF)
option(AGI_THINKING_USE_AVX512 "Use AVX-512 SIMD" OFF)
option(AGI_THINKING_USE_NEON "Use ARM NEON SIMD" OFF)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 CONFIG)

# If pybind11 not found via config, try to fetch it
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

if(AGI_THINKING_USE_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512bw -mavx512dq")
    add_definitions(-DAGI_USE_AVX512)
endif()

if(AGI_THINKING_USE_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    add_definitions(-DAGI_USE_NEON)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

# Source files
set(VSA_SOURCES
    src/bindings.cpp
)

# Build Python module
pybind11_add_module(vsa_core ${VSA_SOURCES})

target_link_libraries(vsa_core PRIVATE
    ${Python3_LIBRARIES}
)

# Install target
install(TARGETS vsa_core
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)

# Tests
if(AGI_THINKING_BUILD_TESTS)
    enable_testing()

    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    # Test executable
    add_executable(test_vsa tests/test_vsa.cpp)
    target_link_libraries(test_vsa PRIVATE GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(test_vsa)
endif()

# Print configuration
message(STATUS "AGI Thinking C++ Extension")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  AVX-512: ${AGI_THINKING_USE_AVX512}")
message(STATUS "  NEON: ${AGI_THINKING_USE_NEON}")
message(STATUS "  Build Tests: ${AGI_THINKING_BUILD_TESTS}")
