cmake_minimum_required(VERSION 3.10)

# PairHMM 单元测试
project(PairHMMTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

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
if(TARGET pairhmm_intra_avx2)
    message(STATUS "Building GoogleTest unit tests")
    
    # 完整的GoogleTest测试（如果AVX512可用）
    if(TARGET pairhmm_intra_avx512)
        add_executable(pairhmm_unittest pairhmm_unittest.cpp)
        target_link_libraries(pairhmm_unittest 
            GTest::gtest 
            GTest::gtest_main
            pairhmm_intra_avx2 
            pairhmm_intra_avx512
        )
        target_compile_options(pairhmm_unittest PRIVATE ${AVX512_FLAGS})
        
        enable_testing()
        add_test(NAME PairHMMGoogleTestFull COMMAND pairhmm_unittest)
        message(STATUS "Added GoogleTest (AVX2 + AVX512)")
    else()
        # 仅AVX2的GoogleTest测试
        add_executable(pairhmm_unittest_avx2 pairhmm_unittest.cpp)
        target_link_libraries(pairhmm_unittest_avx2 
            GTest::gtest 
            GTest::gtest_main
            pairhmm_intra_avx2
        )
        target_compile_options(pairhmm_unittest_avx2 PRIVATE ${AVX2_FLAGS_ONLY})
        
        enable_testing()
        add_test(NAME PairHMMGoogleTestAVX2 COMMAND pairhmm_unittest_avx2)
        message(STATUS "Added GoogleTest (AVX2 only)")
    endif()
else()
    message(WARNING "pairhmm_intra_avx2 target not found, skipping GoogleTest")
endif()

