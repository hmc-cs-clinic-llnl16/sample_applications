add_executable(dependentIndexSets dependentIndexSets.cxx)

target_link_libraries(dependentIndexSets RAJA ${RT_LIBRARIES})

