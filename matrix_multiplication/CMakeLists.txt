if(RAJA_ENABLE_AGENCY)
    set(AGENCY_INCLUDE_DIR ${RAJA_DIR}/extra/agency)
    include_directories(${AGENCY_INCLUDE_DIR})
endif()


if(RAJA_ENABLE_CUDA)
    set_source_files_properties(mmult.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_library(mmult mmult.cu)

    set_source_files_properties(mmult_gpu.cxx PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_executable(mmult.exe mmult_gpu.cxx)
    add_dependencies(mmult.exe mmult RAJA CALIPER)
    target_link_libraries(mmult.exe RAJA mmult CALIPER)
else()
    add_executable(mmult.exe mmult.cxx)
    add_dependencies(mmult.exe RAJA CALIPER)
    target_link_libraries(mmult.exe RAJA CALIPER ${RT_LIBRARIES})
endif()
