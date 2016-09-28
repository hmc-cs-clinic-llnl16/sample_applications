add_executable(mmult.exe mmult.cxx)
add_dependencies(mmult.exe RAJA CALIPER ../common/timing.o)
target_link_libraries(mmult.exe RAJA CALIPER ${RT_LIBRARIES} ../common/timing.o)
