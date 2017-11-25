INCLUDE("../../build-benchmark.cmake")

set(SAC_SRC
  lud.sac
  lud_blocked.sac
)

set (FLAGS -v1 -O3 -noRCO -doPRA -doEXPAR -DSIZE=8192)
BUILD_BENCHMARK ("${FLAGS}")
