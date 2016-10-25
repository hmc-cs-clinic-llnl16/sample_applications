#include <math.h>
#include <complex>
#include <iostream>

#include "RAJA/RAJA.hxx"
#include "fft2d.hxx"

typedef RAJA::omp_reduce reduce_policy;

std::complex<double>* fft2dRajaOmp(double* x, int L, int M) {

  int N = L*M;
  std::complex<double>* bin = new std::complex<double>[N];

  RAJA::forall<RAJA::seq_exec>(0, N, [&](int i) {
    double l = i % L;
    double m = i / L;
    RAJA::ReduceSum<reduce_policy, double> realRowDft(0);
    RAJA::ReduceSum<reduce_policy, double> imagRowDft(0);

    int maxParallelIterations = (RAJA_MAX_REDUCE_VARS-2) / 2;
    int numSaturatedSegments = M / maxParallelIterations;
    int lastSegmentSize = M % maxParallelIterations;
    int numIterationsInSaturatedSegments = M - lastSegmentSize;

    RAJA::IndexSet indexSet;
    RAJA::forall<RAJA::seq_exec>(
        0, numIterationsInSaturatedSegments, maxParallelIterations,
        [&indexSet, maxParallelIterations](int rangeStart) {

      indexSet.push_back(
          RAJA::RangeSegment(rangeStart, rangeStart + maxParallelIterations));
    });

    if (lastSegmentSize > 0) {
      indexSet.push_back(
          RAJA::RangeSegment(
              numIterationsInSaturatedSegments,
              numIterationsInSaturatedSegments + lastSegmentSize));
    }
    indexSet.initDependencyGraph();

    RAJA::forall<RAJA::omp_parallel_for_exec>(
        0, indexSet.getNumSegments() - 1, [&indexSet](int j) {
      RAJA::DepGraphNode* node = indexSet.getSegmentInfo(j)->getDepGraphNode();
      node->numDepTasks() = 1;
      node->depTaskNum(0) = j + 1;
    });

    RAJA::forall<RAJA::omp_parallel_for_exec>(
        1, indexSet.getNumSegments(), [&indexSet](int j) {
      indexSet.getSegmentInfo(j)->getDepGraphNode()->semaphoreValue() = 1;
    });
    indexSet.finalizeDependencyGraph();

    RAJA::forall<RAJA::IndexSet::ExecPolicy<
        RAJA::omp_taskgraph_segit, RAJA::omp_parallel_for_exec>>(
            indexSet, [&](int r) {
      RAJA::ReduceSum<reduce_policy, double> realColDft(0);
      RAJA::ReduceSum<reduce_policy, double> imagColDft(0);

      RAJA::forall<RAJA::omp_parallel_for_exec>(0, L, [&](int q) {
        std::complex<double> colDftTerm(x[q*M + r] * exp(A*((q*l)/L)));
        realColDft += colDftTerm.real();
        imagColDft += colDftTerm.imag();
      });

      std::complex<double> colDft(realColDft, imagColDft);
      std::complex<double> rowDftTerm(exp(A*((r*l)/N)) * colDft * exp(A*((r*m)/M)));
      realRowDft += rowDftTerm.real();
      imagRowDft += rowDftTerm.imag();
    });

    bin[i] = std::complex<double>(realRowDft, imagRowDft);
  });

  return bin;
}
