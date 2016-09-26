add_executable(fftRajaSequential.exe fftRajaSequential.cxx)
add_executable(fftRajaOmp.exe fftRajaOmp.cxx)
add_executable(fftTranspose.exe fftTranspose.cxx)

target_link_libraries(fftTranspose.exe ${RT_LIBRARIES})
target_link_libraries(fftRajaSequential.exe RAJA CALIPER ${RT_LIBRARIES})
target_link_libraries(fftRajaOmp.exe RAJA CALIPER ${RT_LIBRARIES})
