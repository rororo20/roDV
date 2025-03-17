project(rodv LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)

# TensorFlow C++依赖
find_package(TensorFlow REQUIRED)
include_directories(${TENSORFLOW_INCLUDE_DIRS})

# HTSlib集成
add_subdirectory(third_party/htslib)

# 在文件末尾添加以下测试配置
enable_testing()

# 查找Google Test
find_package(GTest REQUIRED)

# 创建测试可执行文件
add_executable(rodv_tests
    tests/test_bam_reader.cpp
    # 添加其他测试文件...
)

# 链接测试依赖项
target_link_libraries(rodv_tests
    PRIVATE
    rodv_core
    GTest::GTest
    GTest::Main
    hts
    ${TENSORFLOW_STATIC_LIBRARIES}
)

# 注册CTest测试用例
add_test(NAME core_tests COMMAND rodv_tests)

add_executable(rodv_main tools/rodv_main.cpp)
target_link_libraries(rodv_main 
    PRIVATE rodv_core 
    ${TENSORFLOW_STATIC_LIBRARIES}
    hts)