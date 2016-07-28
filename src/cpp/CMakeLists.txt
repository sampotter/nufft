project (nufft)
cmake_minimum_required (VERSION 3.4)

set (CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++1z")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DNUFFT_DEBUG")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

find_package (Boost)

add_executable (bookmarks.test bookmarks.test.cpp bookmarks.hpp
  bookmarks.impl.hpp)
add_executable (util.test util.test.cpp util.hpp)
add_executable (index_manip.test index_manip.test.cpp index_manip.hpp
  index_manip.impl.hpp)
add_executable (cauchy.test cauchy.test.cpp cauchy.hpp cauchy.impl.hpp)
add_executable (fmm1d.test fmm1d.test.cpp fmm1d.hpp fmm1d.impl.hpp cauchy.hpp
  cauchy.impl.hpp index_manip.hpp index_manip.impl.hpp bookmarks.hpp
  bookmarks.impl.hpp)
add_executable (nufft.test nufft.test.cpp nufft.hpp nufft.impl.hpp nufft.cpp
  fmm1d.hpp fmm1d.impl.hpp cauchy.hpp cauchy.impl.hpp index_manip.hpp
  index_manip.impl.hpp bookmarks.hpp bookmarks.impl.hpp)

add_executable (fmm1d.perf fmm1d.perf.cpp fmm1d.hpp fmm1d.impl.hpp cauchy.hpp
  cauchy.impl.hpp index_manip.hpp index_manip.impl.hpp bookmarks.hpp
  bookmarks.impl.hpp)
add_executable (nufft.perf nufft.perf.cpp nufft.hpp nufft.impl.hpp nufft.cpp
  cauchy.hpp cauchy.impl.hpp index_manip.hpp index_manip.impl.hpp bookmarks.hpp
  bookmarks.impl.hpp)

add_library (fmm SHARED fmm.hpp fmm.cpp fmm1d.hpp fmm1d.impl.hpp cauchy.hpp
  cauchy.impl.hpp index_manip.hpp index_manip.impl.hpp bookmarks.hpp
  bookmarks.impl.hpp)
add_library (nufft SHARED nufft.hpp nufft.impl.hpp nufft.cpp cauchy.hpp
  cauchy.impl.hpp index_manip.hpp index_manip.impl.hpp bookmarks.hpp
  bookmarks.impl.hpp)

enable_testing ()

add_test (bookmarks bookmarks.test)
add_test (util util.test)
add_test (index_manip index_manip.test)
add_test (cauchy cauchy.test)
add_test (fmm1d fmm1d.test)
add_test (nufft nufft.test)

# Local Variables:
# indent-tabs-mode: nil
# End:
