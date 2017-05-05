
if(RAJA_ENABLE_AGENCY)
    set(AGENCY_INCLUDE_DIR ${RAJA_DIR}/extra/agency)
    include_directories(${AGENCY_INCLUDE_DIR})
endif()


if(RAJA_ENABLE_CUDA)
    set_source_files_properties(main.cxx PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_executable(indexset-bench.exe main.cxx)
    add_dependencies(indexset-bench.exe RAJA CALIPER)
    target_link_libraries(indexset-bench.exe RAJA CALIPER ${RT_LIBRARIES})
else()
    add_executable(indexset-bench.exe main.cxx)
    add_dependencies(indexset-bench.exe CALIPER)
    target_link_libraries(indexset-bench.exe RAJA CALIPER ${RT_LIBRARIES})
endif()
