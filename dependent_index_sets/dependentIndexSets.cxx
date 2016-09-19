
#include "RAJA/RAJA.hxx"
#include "RAJA/exec-openmp/raja_openmp.hxx"
#include <iostream>


int main() {
    // The dependency graph currently only works with range segments
    RAJA::RangeSegment firstSegment(0, 10);
    RAJA::RangeSegment secondSegment(15, 25);

    RAJA::IndexSet indexSet;
    indexSet.push_back(firstSegment);
    indexSet.push_back(secondSegment);

    RAJA::IndexSetSegInfo* firstSegmentInfo = indexSet.getSegmentInfo(0);
    RAJA::IndexSetSegInfo* secondSegmentInfo = indexSet.getSegmentInfo(1);

    firstSegmentInfo->initDepGraphNode();

    // Set the number of tasks (RangeSegments) to be notified when the segment finishes
    firstSegmentInfo->getDepGraphNode()->numDepTasks() = 1;

    // Set the index of the task to be notified; the argument to depTaskNum (in this case 0)
    // is referring to the index of the dependent task within this given segment
    // (i.e., the first task dependent on firstSegmentInfo).  The value (in this case 1)
    // is referring to the index of the dependent range segment within the index set
    // (see above, where secondSegment is at index 1 in the index set)
    firstSegmentInfo->getDepGraphNode()->depTaskNum(0) = 1;

    secondSegmentInfo->initDepGraphNode();

    // Set the number of tasks that must be completed (and signal this task) before this task can start.
    secondSegmentInfo->getDepGraphNode()->semaphoreValue() = 1;

    indexSet.finalizeDependencyGraph();

    std::cout << "The first RAJA forall is not deterministic in what it prints; it does not use the"
              << " dependency graph (see the segment iteration policy)" << std::endl;
    RAJA::forall<RAJA::IndexSet::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::omp_parallel_for_exec>>(
        indexSet,
        [=](int i) {
            std::cout << i << std::endl;
        }
    );

    std::cout << '\n' << '\n' << std::endl;

    std::cout << "The second RAJA forall is deterministic in what it prints because it uses the dependency"
              << " graph, forcing the second range segment to wait for the first one to finish" << std::endl;
    RAJA::forall<RAJA::IndexSet::ExecPolicy<RAJA::omp_taskgraph_segit, RAJA::omp_parallel_for_exec>>(
        indexSet,
        [=](int i) {
            std::cout << i << std::endl;
        }
    );
}
