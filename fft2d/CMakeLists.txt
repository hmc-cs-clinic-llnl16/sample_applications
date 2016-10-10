add_executable(fft2dRajaSequential.exe fft2dRajaSequential.cxx)
add_executable(fft2dRajaOmp.exe fft2dRajaOmp.cxx)

target_link_libraries(fft2dRajaSequential.exe RAJA ${RT_LIBRARIES})
target_link_libraries(fft2dRajaOmp.exe RAJA ${RT_LIBRARIES})
