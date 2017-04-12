add_executable(reducer.exe reducer.cxx)
add_dependencies(fft2d.exe CALIPER)

target_link_libraries(reducer.exe ${RT_LIBRARIES} RAJA CALIPER)
