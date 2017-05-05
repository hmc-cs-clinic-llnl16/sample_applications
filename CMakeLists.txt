cmake_minimum_required(VERSION 3.1)

project(RajaExamples CXX C)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(OpenMP)

set(RAJA_COMPILER "RAJA_COMPILER_GNU" CACHE STRING "")
set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++11 -fext-numeric-literals")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -mavx -std=c++11 -finline-functions -finline-limit=20000" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Ofast -mavx -std=c++11 -finline-functions -finline-limit=20000" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fpermissive" CACHE STRING "")
if (NOT RAJA_ENABLE_CUDA)
    set(RAJA_ENABLE_CUDA Off CACHE BOOL "")
endif()
set(RAJA_ENABLE_AGENCY On CACHE BOOL "")
set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")
set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
set(RAJA_NVCC_FLAGS "-O2 -restrict -arch compute_35 -std c++11 --expt-extended-lambda -ccbin ${CMAKE_CXX_COMPILER} -Xcompiler -fopenmp CACHE LIST \\\"\\\"")

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
    set(RAJA_GIT_REPO https://github.com/hmc-cs-clinic-llnl16/RAJA.git )

  endif()
  ExternalProject_Add(
      raja
      GIT_REPOSITORY ${RAJA_GIT_REPO}
      GIT_TAG ${RAJA_GIT_TAG}
      CMAKE_ARGS 
         -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
         -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
         -DRAJA_ENABLE_CUDA=${RAJA_ENABLE_CUDA}
         -DRAJA_NVCC_FLAGS=${RAJA_NVCC_FLAGS}
         -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
         -DCMAKE_BUILD_TYPE=Release
         -DRAJA_ENABLE_NESTED=On
         -DRAJA_ENABLE_TESTS=Off
         -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
         -DCMAKE_CXX_STANDARD=11
      LOG_DOWNLOAD ON
      LOG_CONFIGURE ON
      LOG_BUILD ON)

  ExternalProject_Get_Property(raja install_dir)
  set(RAJA_DIR ${install_dir}/share/cmake/raja/)
  include_directories(${install_dir}/include)
  if(RAJA_ENABLE_AGENCY)
      set(AGENCY_INCLUDE_DIR ${install_dir}/src/raja/extra/agency)
      include_directories(${AGENCY_INCLUDE_DIR})
  endif()

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
  # This directory may change to /lib/libcaliper.so on some machines
  set_target_properties(CALIPER PROPERTIES
    IMPORTED_LOCATION ${caliper_LIB_DIR}/lib64/libcaliper.so
  )
else ()
  if (NOT CALIPER_GIT_TAG)
    set(CALIPER_GIT_TAG master)
  endif()
  if (NOT CALIPER_GIT_REPO)
    set(CALIPER_GIT_REPO https://github.com/LLNL/Caliper.git )
  endif()
  if (NOT PYTHON_EXECUTABLE)
    set(PYTHON_ADDITIONAL_VERSIONS "2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7")
    find_package(PythonInterp)
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
  # This directory may change to /lib/libcaliper.so on some machines
  set_target_properties(CALIPER PROPERTIES
    IMPORTED_LOCATION ${install_dir}/lib64/libcaliper.so
  )
  add_dependencies(CALIPER caliper)
endif()

add_subdirectory(matrix_multiplication)
add_subdirectory(fft)
add_subdirectory(fft2d)
add_subdirectory(dependent_index_sets)
add_subdirectory(dependent_bench)
add_subdirectory(reducerTests)

set(FIGURES_DIRECTORY "${CMAKE_SOURCE_DIR}/figs")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/caliper_files")

add_custom_command(
  OUTPUT rajaReduce.cali
  COMMAND CALI_CONFIG_PROFILE=thread-trace CALI_LOG_VERBOSITY=2 ./reducer.exe
  COMMAND find *.cali | tail -n 1 | xargs -I '{}' mv '{}' ${CMAKE_BINARY_DIR}/caliper_files/rajaReduce.cali
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/reducerTests
)

add_custom_command(
    OUTPUT reducer_figures
    COMMAND python "${CMAKE_SOURCE_DIR}/analyze_caliper.py" raja reducer
                   -f ${CMAKE_BINARY_DIR}/caliper_files/rajaReduce.cali
                   -o rajaReduce.pdf
                   -q ${CMAKE_BINARY_DIR}/tpl/bin/cali-query
    DEPENDS rajaReduce.cali
)

add_custom_command(
  OUTPUT rajaMmult.cali
  COMMAND CALI_CONFIG_PROFILE=thread-trace CALI_LOG_VERBOSITY=2 ./mmult.exe
  COMMAND find *.cali | tail -n 1 | xargs -I '{}' mv '{}' ${CMAKE_BINARY_DIR}/caliper_files/rajaMmult.cali
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/matrix_multiplication
)
if (NOT RAJA_ENABLE_CUDA)
  add_custom_command(
    OUTPUT mmult_figures
    COMMAND python "${CMAKE_SOURCE_DIR}/analyze_caliper.py" raja mmult
                   -f ${CMAKE_BINARY_DIR}/caliper_files/rajaMmult.cali
                   -o rajaMatrixMultiplication.pdf
                   -q ${CMAKE_BINARY_DIR}/tpl/bin/cali-query
    DEPENDS rajaMmult.cali
  )
else()
  add_custom_command(
    OUTPUT mmult_figures
    COMMAND python "${CMAKE_SOURCE_DIR}/analyze_caliper.py" raja mmultgpu
                   -f ${CMAKE_BINARY_DIR}/caliper_files/rajaMmult.cali
                   -o rajaMatrixMultiplication
                   -q ${CMAKE_BINARY_DIR}/tpl/bin/cali-query
    DEPENDS rajaMmult.cali
  )
endif()

add_custom_command(
  OUTPUT rajafft1d.cali
  COMMAND CALI_CONFIG_PROFILE=thread-trace CALI_LOG_VERBOSITY=2 ./fftTranspose.exe
  COMMAND find *.cali | tail -n 1 | xargs -I '{}' mv '{}' ${CMAKE_BINARY_DIR}/caliper_files/rajafft1d.cali
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/fft
)

add_custom_command(
  OUTPUT 1dfft_figures
  COMMAND python "${CMAKE_SOURCE_DIR}/analyze_caliper.py" raja fft
                 -f ${CMAKE_BINARY_DIR}/caliper_files/rajafft1d.cali
                 -o rajaFft1D
                 -q ${CMAKE_BINARY_DIR}/tpl/bin/cali-query
  DEPENDS rajafft1d.cali
)

add_custom_command(
  OUTPUT rajafft2d.cali
  COMMAND CALI_CONFIG_PROFILE=thread-trace CALI_LOG_VERBOSITY=2 ./fft2d.exe
  COMMAND find *.cali | tail -n 1 | xargs -I '{}' mv '{}' ${CMAKE_BINARY_DIR}/caliper_files/rajafft2d.cali
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/fft2d
)

add_custom_command(
  OUTPUT 2dfft_figures
  COMMAND python "${CMAKE_SOURCE_DIR}/analyze_caliper.py" raja fft2d
                 -f ${CMAKE_BINARY_DIR}/caliper_files/rajafft2d.cali
                 -o rajaFft2D
                 -q ${CMAKE_BINARY_DIR}/tpl/bin/cali-query
  DEPENDS rajafft2d.cali
)

add_custom_command(
  OUTPUT dependentBench.cali
  COMMAND CALI_CONFIG_PROFILE=thread-trace CALI_LOG_VERBOSITY=2 ./indexset-bench.exe
  COMMAND find *.cali | tail -n 1 | xargs -I '{}' mv '{}' ${CMAKE_BINARY_DIR}/caliper_files/dependent_bench.cali
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/dependent_bench
)

add_custom_command(
  OUTPUT dependentBench_figures
  COMMAND python "${CMAKE_SOURCE_DIR}/analyze_caliper.py" raja dependent_bench
                 -f ${CMAKE_BINARY_DIR}/caliper_files/dependent_bench.cali
                 -o dependent_bench
                 -q ${CMAKE_BINARY_DIR}/tpl/bin/cali-query
  DEPENDS dependentBench.cali
)

add_custom_target(figures DEPENDS
  dependentBench_figures
  2dfft_figures
  reducer_figures
  1dfft_figures
  mmult_figures
)

