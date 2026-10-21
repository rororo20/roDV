cmake_minimum_required(VERSION 3.10)

find_package(GTest REQUIRED)

add_executable(inference_unittest
    preprocess_test.cpp
    launch_plan_test.cpp)
target_link_libraries(inference_unittest
    GTest::gtest
    GTest::gtest_main
    inference)
add_test(NAME InferenceTest COMMAND inference_unittest)

add_executable(preprocess_perftest preprocess_perftest.cpp)
target_link_libraries(preprocess_perftest inference)
