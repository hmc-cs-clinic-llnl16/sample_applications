if(RAJA_ENABLE_AGENCY)
    set(AGENCY_INCLUDE_DIR ${RAJA_DIR}/extra/agency)
    include_directories(${AGENCY_INCLUDE_DIR})
endif()

if(RAJA_ENABLE_CUDA)
    set_source_files_properties(fft2d.cxx PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_executable(fft2d.exe fft2d.cxx) 
else()
    add_executable(fft2d.exe fft2d.cxx) 
endif()
add_dependencies(fft2d.exe RAJA CALIPER)
target_link_libraries(fft2d.exe RAJA CALIPER)

