if(RAJA_ENABLE_CUDA)
    set_source_files_properties(dependentIndexSets.cxx PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_executable(dependentIndexSets dependentIndexSets.cxx)
else()
    add_executable(dependentIndexSets dependentIndexSets.cxx)
endif()

target_link_libraries(dependentIndexSets RAJA ${RT_LIBRARIES})

