INCLUDE("../../build-benchmark.cmake")

set(SAC_SRC
  srad.sac
)

set (FLAGS -v1 -O3 -noRCO -DSIZE=2048 -DITER=1000)
BUILD_BENCHMARK ("${FLAGS}")
