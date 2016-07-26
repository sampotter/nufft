project (nfft_timer)
cmake_minimum_required (VERSION 3.4)

set (CMAKE_CXX_FLAGS "-Wall -Wextra -O4 -std=c++11")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

set (CMAKE_C_FLAGS "-Wall -Wextra -O4 -std=c99")
set (CMAKE_C_FLAGS_DEBUG "-O0 -g -DDEBUG")
set (CMAKE_C_FLAGS_RELEASE "-O3")
set (CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g")

add_library (nfft_wrapper src/nfft_wrapper.c src/nfft_wrapper.h)
target_link_libraries (nfft_wrapper nfft3)

add_library (nfft_timer SHARED src/nfft_timer.cpp src/nfft_timer.hpp)
target_link_libraries (nfft_timer nfft_wrapper)
