add_executable(reducer.exe reducer.cxx agencyReducer.hxx)
add_dependencies(reducer.exe CALIPER)

target_link_libraries(reducer.exe ${RT_LIBRARIES} RAJA CALIPER)
