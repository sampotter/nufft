project (nufft)
cmake_minimum_required (VERSION 3.4)

set (CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++1z")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DNUFFT_DEBUG")
set (CMAKE_CXX_FLAGS_RELEASE "-O4")
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

find_package (Boost)

add_executable (util.test util.test.cpp)
add_executable (index_manip.test index_manip.test.cpp index_manip.cpp)

enable_testing ()

add_test (util util.test)
add_test (index_manip index_manip.test)
