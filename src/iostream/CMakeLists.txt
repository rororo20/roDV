

file(GLOB ALL_SRC "*.cpp")

add_library(iostream ${ALL_SRC})

target_link_libraries(iostream PRIVATE ${HTSLIB_LIBRARY})

include_directories(${HTSLIB_INCLUDE_DIR})

install(TARGETS iostream DESTINATION lib)