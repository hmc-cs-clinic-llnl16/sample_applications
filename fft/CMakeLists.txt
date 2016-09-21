add_executable(fftRajaSequential.exe fftRajaSequential.cxx)
add_executable(fftRajaOmp.exe fftRajaSequential.cxx)
add_executable(fftTranspose.exe fftTranspose.cxx)

target_link_libraries(fftRajaSequential.exe ${RT_LIBRARIES})
target_link_libraries(fftRajaOmp.exe ${RT_LIBRARIES})
target_link_libraries(fftTranspose.exe ${RT_LIBRARIES})