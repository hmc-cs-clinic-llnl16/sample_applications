if(RAJA_ENABLE_AGENCY)
    set(AGENCY_INCLUDE_DIR ${RAJA_DIR}/extra/agency)
    include_directories(${AGENCY_INCLUDE_DIR})
endif()

if(RAJA_ENABLE_CUDA)
    set_source_files_properties(reducer.cxx PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_executable(reducer.exe reducer.cxx agencyReducer.hxx) 
else()
    add_executable(reducer.exe reducer.cxx agencyReducer.hxx) 
endif()


add_dependencies(reducer.exe CALIPER)

target_link_libraries(reducer.exe ${RT_LIBRARIES} RAJA CALIPER)
