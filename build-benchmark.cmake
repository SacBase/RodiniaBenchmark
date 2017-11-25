# This file is a macro that can be used to build the
# sac rodinia benchmarks.

# Assumptions:
#   *SAC_SRC has been previously defined.  

# This macro builds rodinina benchmarks with the specified flags (if any).
#
#
# Arguments:
#       sac2c_flags:
#               Flags that sac2c needs to build the program.
MACRO (BUILD_BENCHMARK sac2c_flags)
    # Sanity check. SAC_SRC 
    # should hold the list of sac files needed to compile
    # the current benchmark and should be set in the CMakeLists.txt
    # file located in the root of each benchmark.
    IF(NOT SAC_SRC) 
        MESSAGE(FATAL_ERROR, "The sac file dependencies variable is not set")
    ENDIF()
    
    INCLUDE ("${CMAKE_SOURCE_DIR}/cmake-common/sac2c-variables.cmake")
    INCLUDE ("${CMAKE_SOURCE_DIR}/cmake-common/build-sac2c-program.cmake")
    INCLUDE ("${CMAKE_SOURCE_DIR}/cmake-common/resolve-sac2c-dependencies.cmake")
    
    FOREACH(name ${SAC_SRC})
      SET (src "${CMAKE_CURRENT_SOURCE_DIR}/${name}")
    
      GET_FILENAME_COMPONENT (dir ${name} DIRECTORY)
    
      GET_FILENAME_COMPONENT (dst ${name} NAME_WE)
    
      FILE (MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${dir}")

      #SAC2C_COMPILE_PROG is a macro located in cmake-common/build-sac2c-program.cmake
      SAC2C_COMPILE_PROG(${name} ${dst}.out "${SAC_SRC}" "${sac2c_flags}")
    
    ENDFOREACH(name)
ENDMACRO()
