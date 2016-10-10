


add_executable(fft fft.cxx)
add_dependencies(fft CALIPER)
include_directories(/clinic/2016/llnl16/test/agency)
target_link_libraries(fft CALIPER ${RT_LIBRARIES})