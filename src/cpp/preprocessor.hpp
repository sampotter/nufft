#ifndef __NUFFT_PREPROCESSOR_HPP__
#define __NUFFT_PREPROCESSOR_HPP__

// We define the following two preprocessor symbols (NUFFT_PRIVATE
// and NUFFT_PROTECTED) in order to make it possible to directly
// access any fields or functions of a class or struct while unit
// testing.

#ifdef NUFFT_UNIT_TEST
#    define NUFFT_PRIVATE public
#else
#    define NUFFT_PRIVATE private
#endif

#ifdef NUFFT_UNIT_TEST
#    define NUFFT_PROTECTED public
#else
#    define NUFFT_PROTECTED protected
#endif

#endif // __NUFFT_PREPROCESSOR_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
