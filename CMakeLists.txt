cmake_minimum_required(VERSION 3.16)
project(rodv LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)

# 测试配置
enable_testing()
find_package(GTest REQUIRED)

# 测试可执行文件
add_executable(rodv_tests
    tests/test_bam_reader.cpp
    # 添加其他测试文件...
)

# 测试目标链接
target_link_libraries(rodv_tests
    PRIVATE
    rodv_core
    GTest::GTest
    GTest::Main
    hts
)

# 注册测试用例
add_test(NAME core_tests COMMAND rodv_tests)

# 主程序可执行文件
add_executable(rodv_main tools/rodv_main.cpp)
target_link_libraries(rodv_main 
    PRIVATE 
    rodv_core 
    hts)
