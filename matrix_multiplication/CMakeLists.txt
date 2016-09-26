add_executable(mmult.exe mmult.cxx ../common/timing.cxx)
add_dependencies(mmult.exe RAJA CALIPER)
target_link_libraries(mmult.exe RAJA CALIPER ${RT_LIBRARIES}
  ../common/timing.cxx)
