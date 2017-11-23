cmake_minimum_required(VERSION 3.5)
project(demo)

SET (DLL_BUILD_DIR  "${PROJECT_BINARY_DIR}/benchmarks")
FILE(MAKE_DIRECTORY ${DLL_BUILD_DIR})

SET (TARGETS            seq mt_pth  CACHE STRING "Build stdlib-jpeg for these targets")
SET (SAC2C_EXEC                     CACHE STRING "A path to sac2c compiler")
SET (LINKSETSIZE        "500"       CACHE STRING "Set a value for -linksetsize parameter of sac2c")

include("cmake-common/check-sac2c.cmake")

#set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# For every target run CMakeLists.txt in src
SET (TARGET_COUNT 1)
FOREACH (TARGET IN ITEMS ${TARGETS})
    #MESSAGE ("target #${TARGET_COUNT} = ${TARGET}")
    ADD_SUBDIRECTORY (benchmarks benchmarks/benchmarks-${TARGET})
    MATH (EXPR TARGET_COUNT "${TARGET_COUNT} + 1")
ENDFOREACH ()

#add_executable(hello ${PROJECT_SOURCE_DIR}/test.sac)
