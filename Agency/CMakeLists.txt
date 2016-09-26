add_executable(mmultAgency.exe mmultAgency.cpp)
add_dependencies(mmultAgency.exe CALIPER)
include_directories(/clinic/2016/llnl16/test/agency)
target_link_libraries(mmultAgency.exe CALIPER ${RT_LIBRARIES})

