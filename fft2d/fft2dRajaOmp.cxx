#include <math.h>
#include <complex>
#include <iostream>

#include "RAJA/RAJA.hxx"

typedef RAJA::omp_reduce reduce_policy;

const double PI = acos(-1);
const std::complex<double> A = -2*PI*1i;

std::complex<double>* fft2d(double* x, int L, int M) {

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

int main() {

    //The number of samples which were collected(must be a power of 2)
    int numberSamples = pow(2, 8);
    //The rate at which the samples were taken in Hz
    double sampleRate = 40.0;

    //make sure this is less than half of sample rate
    double inputFreq = 10.0;

    //make the input array
    double sine[numberSamples];
    for (int i = 0; i < numberSamples; ++i)
    {
        sine[i] = 20*sin(2*PI * inputFreq * (i / sampleRate));
    }

    //get the bins from the fft
    int L = 4;
    int M = pow(2, 6);
    std::complex<double>* bins = fft2d(sine, L, M);
    
    //Print out the results of the fft but adjust the numbers so they correspond 
    for (int i = 0; i < numberSamples; ++i){
        std::cout << (sampleRate/numberSamples)*i <<": " << std::abs(bins[i]) << std::endl;
    }

    return 0;
}
