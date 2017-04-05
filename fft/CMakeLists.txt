if(RAJA_ENABLE_AGENCY)
    set(AGENCY_INCLUDE_DIR ${RAJA_DIR}/extra/agency)
    include_directories(${AGENCY_INCLUDE_DIR})
endif()

add_executable(fftRajaSequential.exe fftRajaSequential.cxx)
add_executable(fftRajaOmp.exe fftRajaSequential.cxx)

target_link_libraries(fftRajaSequential.exe ${RT_LIBRARIES})
target_link_libraries(fftRajaOmp.exe ${RT_LIBRARIES})
