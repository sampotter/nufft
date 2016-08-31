project (nufft)
cmake_minimum_required (VERSION 3.4)

set (CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++1z")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DNUFFT_DEBUG")
set (CMAKE_CXX_FLAGS_RELEASE "-O3 -flto -ffast-math -DNUFFT_RELEASE")
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -ffast-math -g -DNUFFT_RELEASE")

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
add_executable (math.test math.test.cpp math.hpp)
add_executable (nufft.test nufft.test.cpp nufft.hpp nufft.impl.hpp nufft.cpp
  fmm1d.hpp fmm1d.impl.hpp cauchy.hpp cauchy.impl.hpp index_manip.hpp
  index_manip.impl.hpp bookmarks.hpp bookmarks.impl.hpp)
add_executable (source_coefs.test source_coefs.test.cpp source_coefs.hpp
  source_coefs.impl.hpp)
add_executable (stencils.test stencils.test.cpp stencils.hpp stencils.impl.hpp)

set (nufft_tests
  bookmarks
  util
  index_manip
  cauchy
  fmm1d
  math
  nufft
  source_coefs
  stencils)
foreach (test ${nufft_tests})
  set_target_properties (${test}.test PROPERTIES COMPILE_FLAGS
    "-DNUFFT_UNIT_TEST")
  add_test (${test} ${test}.test)
endforeach ()

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

# Local Variables:
# indent-tabs-mode: nil
# End:
