cmake_minimum_required(VERSION 3.10)

# PairHMM 单元测试
project(PairHMMTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# AVX2 编译标志
set(AVX2_FLAGS_ONLY "-mavx2;-mfma")

# AVX512 编译标志
set(AVX512_FLAGS "-mavx512f;-mavx512dq;-mavx512bw;-mavx512vl;-mavx2;-mfma")

# 头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# GoogleTest 配置
find_package(GTest REQUIRED)

# ============================================================================
# GoogleTest 单元测试
# ============================================================================

# GoogleTest 单元测试（AVX2 + AVX512）
message(STATUS "Building GoogleTest unit tests")

# GoogleTest测试，链接AVX2和AVX512库
add_executable(pairhmm_unittest pairhmm_unittest.cpp)
target_link_libraries(pairhmm_unittest 
    GTest::gtest 
    GTest::gtest_main
    pairhmm_intra_avx2 
    pairhmm_intra_avx512
    pairhmm_inter_avx2
    pairhmm_inter_avx512
)
target_compile_options(pairhmm_unittest PRIVATE ${AVX512_FLAGS})

# Schedule 调度测试
add_executable(pairhmm_schedule_unittest pairhmm_schedule_unittest.cpp)
target_link_libraries(pairhmm_schedule_unittest 
    GTest::gtest 
    GTest::gtest_main
    pairhmm_schedule
    pairhmm_intra_avx2 
    pairhmm_intra_avx512
    pairhmm_inter_avx2
    pairhmm_inter_avx512
)
target_compile_options(pairhmm_schedule_unittest PRIVATE ${AVX512_FLAGS})
add_test(NAME PairHMMScheduleTest COMMAND pairhmm_schedule_unittest)
message(STATUS "Added Schedule GoogleTest (AVX2 + AVX512)")

# Assemble日志解析测试
add_executable(pairhmm_unitest_from_assemble pairhmm_unitest_from_assemble.cpp)
target_link_libraries(pairhmm_unitest_from_assemble 
    GTest::gtest 
    GTest::gtest_main
    pairhmm_schedule
    pairhmm_intra_avx2 
    pairhmm_intra_avx512
    pairhmm_inter_avx2
    pairhmm_inter_avx512
)
target_compile_options(pairhmm_unitest_from_assemble PRIVATE ${AVX512_FLAGS})
add_test(NAME PairHMMAssembleLogTest COMMAND pairhmm_unitest_from_assemble)
message(STATUS "Added AssembleLog GoogleTest (AVX2 + AVX512)")

enable_testing()
add_test(NAME PairHMMGoogleTest COMMAND pairhmm_unittest)
message(STATUS "Added GoogleTest (AVX2 + AVX512)")

