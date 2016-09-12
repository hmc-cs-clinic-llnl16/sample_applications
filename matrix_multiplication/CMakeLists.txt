add_executable(mmult.exe mmult.cxx)

target_link_libraries(mmult.exe ${RT_LIBRARIES} ${CALIPER_LIBRARIES})
