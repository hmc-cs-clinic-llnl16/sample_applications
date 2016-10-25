add_executable(fft2d.exe fft2d.cxx fft2dRajaSequential.cxx fft2dRajaOmp.cxx fft2dSequential.cxx)
add_dependencies(fft2d.exe CALIPER)
target_link_libraries(fft2d.exe RAJA CALIPER ${RT_LIBRARIES})

