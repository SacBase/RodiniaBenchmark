INCLUDE("../../build-benchmark.cmake")

set(SAC_SRC
  pathfinder.sac
)

set (FLAGS -v1 -O3 -noRCO -DSIZE=512000 -DITER=100)
BUILD_BENCHMARK ("${FLAGS}")
