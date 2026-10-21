cmake_minimum_required(VERSION 3.10)

find_package(GTest REQUIRED)

add_executable(make_examples_unittest
    snp_counter_test.cpp
    region_pipeline_test.cpp
    realigner_test.cpp)
target_link_libraries(make_examples_unittest
    GTest::gtest
    GTest::gtest_main
    make_examples)
add_test(NAME MakeExamplesSnpTest COMMAND make_examples_unittest)
