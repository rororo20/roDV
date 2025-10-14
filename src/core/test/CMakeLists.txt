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

# ============================================================================
# 根据编译的库决定测试目标
# ============================================================================

# AVX2 测试（始终编译，如果 AVX2 库存在）
if(TARGET pairhmm_intra_avx2)
    message(STATUS "Building AVX2 test")
    add_executable(pairhmm_test_avx2 pairhmm_test_avx2.cpp)
    target_link_libraries(pairhmm_test_avx2 pairhmm_intra_avx2)
    target_compile_options(pairhmm_test_avx2 PRIVATE ${AVX2_FLAGS_ONLY})
    
    enable_testing()
    add_test(NAME PairHMMUnitTestAVX2 COMMAND pairhmm_test_avx2)
endif()

# 完整测试（仅在 AVX512 可用时编译）
if(TARGET pairhmm_intra_avx512)
    message(STATUS "Building full test (AVX2 + AVX512)")
    add_executable(pairhmm_test pairhmm_test.cpp)
    target_link_libraries(pairhmm_test pairhmm_intra_avx2 pairhmm_intra_avx512)
    target_compile_options(pairhmm_test PRIVATE ${AVX512_FLAGS})
    
    add_test(NAME PairHMMUnitTestFull COMMAND pairhmm_test)
else()
    message(STATUS "AVX512 not available, skipping full test")
endif()
