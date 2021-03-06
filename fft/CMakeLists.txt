if(RAJA_ENABLE_AGENCY)
    set(AGENCY_INCLUDE_DIR ${RAJA_DIR}/extra/agency)
    include_directories(${AGENCY_INCLUDE_DIR})
endif()

if(RAJA_ENABLE_CUDA)
    set_source_files_properties(fftRajaSequential.cxx PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_executable(fftRajaSequential.exe fftRajaSequential.cxx)
    add_dependencies(fftRajaSequential.exe RAJA CALIPER)
    target_link_libraries(fftRajaSequential.exe RAJA CALIPER)

    set_source_files_properties(fftRajaOmp.cxx PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_executable(fftRajaOmp.exe fftRajaOmp.cxx)
    add_dependencies(fftRajaOmp.exe RAJA CALIPER)
    target_link_libraries(fftRajaOmp.exe RAJA CALIPER)

    set_source_files_properties(fftTranspose.cxx PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_executable(fftTranspose.exe fftTranspose.cxx)
    add_dependencies(fftTranspose.exe RAJA CALIPER)
    target_link_libraries(fftTranspose.exe RAJA CALIPER)
else()
    add_executable(fftRajaSequential.exe fftRajaSequential.cxx)
    add_executable(fftRajaOmp.exe fftRajaOmp.cxx)
    add_executable(fftTranspose.exe fftTranspose.cxx)

    target_link_libraries(fftTranspose.exe RAJA CALIPER ${RT_LIBRARIES})
    target_link_libraries(fftRajaSequential.exe RAJA CALIPER ${RT_LIBRARIES})
    target_link_libraries(fftRajaOmp.exe RAJA CALIPER ${RT_LIBRARIES})
endif()
