if(RAJA_ENABLE_AGENCY)
    set(AGENCY_INCLUDE_DIR ${RAJA_DIR}/extra/agency)
    include_directories(${AGENCY_INCLUDE_DIR})
endif()

add_executable(mmult.exe mmult.cxx)
add_dependencies(mmult.exe RAJA CALIPER)
target_link_libraries(mmult.exe RAJA CALIPER ${RT_LIBRARIES})
