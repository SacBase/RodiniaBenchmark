INCLUDE("../../build-benchmark.cmake")
set(SAC_SRC
    backprop.sac
    backprop_no_boundary.sac
)

set(FLAGS -v1 -O3 -noRCO -DFLOAT)
BUILD_BENCHMARK("${FLAGS}")
