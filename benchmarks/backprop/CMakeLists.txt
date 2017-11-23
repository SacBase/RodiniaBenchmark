cmake_minimum_required(VERSION 3.5)
INCLUDE ("${CMAKE_SOURCE_DIR}/cmake-common/sac2c-variables.cmake")
INCLUDE ("${CMAKE_SOURCE_DIR}/cmake-common/build-sac2c-program.cmake")
INCLUDE ("${CMAKE_SOURCE_DIR}/cmake-common/resolve-sac2c-dependencies.cmake")

set(SAC_SRC
    backprop.sac
    backprop_no_boundary.sac
)

FOREACH(name ${SAC_SRC})
  SET (src "${CMAKE_CURRENT_SOURCE_DIR}/${name}")

  GET_FILENAME_COMPONENT (dir ${name} DIRECTORY)

  GET_FILENAME_COMPONENT (dst ${name} NAME_WE)

  SET (dst "${CMAKE_CURRENT_BINARY_DIR}/${dir}/${dst}.out")

  FILE (MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${dir}")

  MESSAGE(STATUS "src: ${src}")
  MESSAGE(STATUS "dst: ${dst}")
  MESSAGE(STATUS "sac_src: ${SAC_SRC}")
  MESSAGE(STATUS "make_directory: ${CMAKE_CURRENT_BINARY_DIR}/${dir}")

  SAC2C_COMPILE_PROG(${name} ${name}.out "${SAC_SRC}" "")

ENDFOREACH(name)
