cmake_minimum_required(VERSION 3.10)

add_library(inception_preprocess_scalar STATIC preprocess_scalar.cpp)
target_include_directories(inception_preprocess_scalar PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

add_library(inception_preprocess_avx2 STATIC preprocess_avx2.cpp)
target_include_directories(inception_preprocess_avx2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(inception_preprocess_avx2 PRIVATE -mavx2 -mfma)
target_link_libraries(inception_preprocess_avx2 PUBLIC inception_preprocess_scalar)

add_library(inception_preprocess_avx512 STATIC preprocess_avx512.cpp)
target_include_directories(inception_preprocess_avx512 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(inception_preprocess_avx512 PRIVATE
    -mavx512f -mavx512cd -mavx512dq -mavx512bw -mavx512vl -mavx2 -mfma)
target_link_libraries(inception_preprocess_avx512 PUBLIC inception_preprocess_scalar)

add_library(inference STATIC
    preprocess_dispatch.cpp
    launch_plan.cpp)
target_include_directories(inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(inference PUBLIC
    inception_preprocess_avx2
    inception_preprocess_avx512)

add_subdirectory(test)
