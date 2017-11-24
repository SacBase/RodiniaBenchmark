INCLUDE("../../build-benchmark.cmake")

set(SAC_SRC
  hotspot.sac
)

SET (FLAGS -v1 -O3 -noRCO -DSIZE=1024 -DITER=20)
BUILD_BENCHMARK ("${FLAGS}")
