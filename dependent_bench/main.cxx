
#include "RAJA/RAJA.hxx"
#include "caliper/Annotation.h"
#include <algorithm>


RAJA::IndexSet createDepIndexSet(const int SEGMENTS)
{
	RAJA::IndexSet indexSet;

	for (int i = 0; i < SEGMENTS; ++i) {
		indexSet.push_back(RAJA::RangeSegment(i, i + 1));
	}

	indexSet.initDependencyGraph();
	for (int i = 0; i < SEGMENTS; ++i) {
		RAJA::DepGraphNode* node = indexSet.getSegmentInfo(i)->getDepGraphNode();
		if (i < SEGMENTS - 1) {
     		node->numDepTasks() = 1;
      		node->depTaskNum(0) = i + 1;
    	}
    	if (i > 0) {
      		node->semaphoreValue() = 1;
    	}
  	}
 	indexSet.finalizeDependencyGraph();

	return indexSet;
}

RAJA::IndexSet createManyDepIndexSet(const int SEGMENTS)
{
    RAJA::IndexSet indexSet;

    for (int i = 0; i < SEGMENTS; ++i) {
        indexSet.push_back(RAJA::RangeSegment(i, i + 1));
    }

    indexSet.initDependencyGraph();
    for (int i = 0; i < SEGMENTS; ++i) {
        RAJA::DepGraphNode* node = indexSet.getSegmentInfo(i)->getDepGraphNode();
        if (i > 0) {
            node->numDepTasks() = 1;
            node->depTaskNum(0) = i - 1;
        }
        if (i < SEGMENTS - 1) {
            node->semaphoreValue() = 1;
        }
    }
    indexSet.finalizeDependencyGraph();
    return indexSet;
}

template <typename ExecPolicy>
void run_benchmark(RAJA::IndexSet& indexSet, int* result) {

	RAJA::forall<ExecPolicy>(indexSet, [&](int i) {
		result[i] = result[std::max(0, i - 1) + i];
	});
}

struct AgencyPolicy {
	using EXEC = typename RAJA::IndexSet::ExecPolicy<RAJA::agency_taskgraph_parallel_segit, RAJA::seq_exec>;
	constexpr static const char* name = "Agency";
};

struct OmpPolicy {
	using EXEC = typename RAJA::IndexSet::ExecPolicy<RAJA::omp_taskgraph_segit, RAJA::seq_exec>;
	constexpr static const char* name = "OMP";
};

int main() {
    const int num_iters = 10;
    const int MIN_SIZE = 1000;
    const int MAX_SIZE = 1000000;
	cali::Annotation iteration = cali::Annotation("iteration");
    cali::Annotation benchmarkSize = cali::Annotation("size");
    cali::Annotation loopIter = cali::Annotation("loop");    


    for (int size = MIN_SIZE; size <= MAX_SIZE; size *= 10) {
        benchmarkSize.set(size);

        for (int i = 0; i < num_iters; ++i) {
            loopIter.set(i);
            std::vector<int> results;
            results.reserve(size);

    		RAJA::IndexSet agencyIndexSet = createDepIndexSet(size);
    		RAJA::IndexSet ompIndexSet = createDepIndexSet(size);
    		
    		iteration.set(AgencyPolicy::name);
    		run_benchmark<AgencyPolicy::EXEC>(agencyIndexSet, results.data());
    
    		iteration.set(OmpPolicy::name);
    		run_benchmark<OmpPolicy::EXEC>(ompIndexSet, results.data());
    
    		iteration.set("garbage");
            std::cout << results[10] << '\n';
        }
    }
    //benchmarkSize.end();
	//iteration.end();
    //loopIter.end();
}
