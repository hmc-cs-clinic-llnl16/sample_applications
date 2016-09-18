cmake_minimum_required(VERSION 3.1)

project(RajaExamples CXX C)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include(ExternalProject)
SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/tpl)

if (RAJA_DIR)
  find_package(RAJA QUIET)
  include_directories(${RAJA_INCLUDE_DIR})
else ()
  if (NOT RAJA_GIT_TAG)
    set(RAJA_GIT_TAG develop)
  endif()
  if (NOT RAJA_GIT_REPO)
    set(RAJA_GIT_REPO https://github.com/LLNL/RAJA.git )
  endif()
  ExternalProject_Add(
      raja
      GIT_REPOSITORY ${RAJA_GIT_REPO}
      GIT_TAG ${RAJA_GIT_TAG}
      CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DRAJA_ENABLE_CUDA=${RAJA_ENABLE_CUDA}
        -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DRAJA_ENABLE_NESTED=On
        -DRAJA_ENABLE_TESTS=Off
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/tpl
      LOG_DOWNLOAD ON
      LOG_CONFIGURE ON
      LOG_BUILD ON)

  ExternalProject_Get_Property(raja install_dir)
  set(RAJA_DIR ${install_dir}/share/cmake/raja/)
  include_directories(${install_dir}/include)

  add_library(RAJA      UNKNOWN IMPORTED)
  set_target_properties(RAJA PROPERTIES
    IMPORTED_LOCATION ${install_dir}/lib/libRAJA.a
  )
  add_dependencies(RAJA      raja)
endif()

if(RAJA_ENABLE_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "CUDA Enabled")
    set (CUDA_NVCC_FLAGS ${RAJA_NVCC_FLAGS})
    set (CUDA_PROPAGATE_HOST_FLAGS OFF)
    include_directories(${CUDA_INCLUDE_DIRS})
  endif()
endif()

if (CALIPER_DIR)
  find_package(CALIPER QUIET)
  include_directories(${caliper_INCLUDE_DIR})
  add_library(CALIPER UNKNOWN IMPORTED)
  set_target_properties(CALIPER PROPERTIES
    IMPORTED_LOCATION ${caliper_LIB_DIR}/lib/libcaliper.so
  )
else ()
  if (NOT CALIPER_GIT_TAG)
    set(CALIPER_GIT_TAG master)
  endif()
  if (NOT CALIPER_GIT_REPO)
    set(CALIPER_GIT_REPO https://github.com/LLNL/Caliper.git )
  endif()
  ExternalProject_Add(
      caliper
      GIT_REPOSITORY ${CALIPER_GIT_REPO}
      GIT_TAG ${CALIPER_GIT_TAG}
      CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/tpl
        -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      LOG_DOWNLOAD ON
      LOG_CONFIGURE ON
      LOG_BUILD ON)

  ExternalProject_Get_Property(caliper install_dir)
  set(CALIPER_DIR ${install_dir}/share/cmake/caliper/)
  include_directories(${install_dir}/include)

  add_library(CALIPER      UNKNOWN IMPORTED)
  set_target_properties(CALIPER PROPERTIES
    IMPORTED_LOCATION ${install_dir}/lib/libcaliper.so
  )
  add_dependencies(CALIPER caliper)
endif()

add_subdirectory(matrix_multiplication)
add_subdirectory(fft)

