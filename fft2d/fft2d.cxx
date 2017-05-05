#include <math.h>

#include <array>
#include <complex>
#include <iostream>

#include "RAJA/RAJA.hxx"
#include "caliper/Annotation.h"

const double PI = acos(-1);
const std::complex<double> A(0, -2*PI);

std::complex<double>* fft2dSequential(double* x, int L, int M) {

  int N = L*M;
  std::complex<double>* bin = new std::complex<double>[N];

  for (int i = 0; i < N; ++i) {
    double l = i % L;
    double m = i / L;
    std::complex<double> rowDft = 0;
    for (int r = 0; r < M; ++r) {
      std::complex<double> colDft = 0;
      for (int q = 0; q < L; ++q) {
        colDft += x[q*M + r] * exp(A*((q*l)/L));
      }
      rowDft += exp(A*((r*l)/N)) * colDft * exp(A*((r*m)/M));
    }

    bin[i] = rowDft;
  }
  return bin;
}

std::complex<double>* fft2dRajaSequential(double* x, int L, int M) {
  typedef RAJA::seq_exec exec_policy;
  typedef RAJA::seq_reduce reduce_policy;

  int N = L*M;
  std::complex<double>* bin = new std::complex<double>[N];

  RAJA::forall<exec_policy>(0, N, [&](int i) {
    double l = i % L;
    double m = i / L;
    RAJA::ReduceSum<reduce_policy, double> realRowDft(0);
    RAJA::ReduceSum<reduce_policy, double> imagRowDft(0);

    RAJA::forall<exec_policy>(0, M, [&](int r) {
      RAJA::ReduceSum<reduce_policy, double> realColDft(0);
      RAJA::ReduceSum<reduce_policy, double> imagColDft(0);

      RAJA::forall<exec_policy>(0, L, [&](int q) {
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

std::complex<double>* fft2dRajaOmp(double* x, int L, int M) {
  using reduce_policy = RAJA::omp_reduce;

  int N = L*M;
  std::complex<double>* bin = new std::complex<double>[N];

  RAJA::forall<RAJA::seq_exec>(0, N, [&](int i) {
    double l = i % L;
    double m = i / L;
    RAJA::ReduceSum<reduce_policy, double> realRowDft(0);
    RAJA::ReduceSum<reduce_policy, double> imagRowDft(0);

    int maxParallelIterations = (RAJA_MAX_REDUCE_VARS-2) / 2;
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

void checkResult(const std::complex<double> * actual, const std::complex<double> * expected, const int size) {
  for (int i = 0; i < size; ++i) {
    if (actual[i] != expected[i]) {
      std::ostringstream os;
      os << "Invalid value at: " << i << ". Was " << actual[i] << " but expected " << expected[i] << "\n";
      throw std::runtime_error(os.str());
    }
  }
}

template <typename Functor>
void runTimingTest(Functor f, double* x, int L, int M, int numTrials, const std::complex<double> * expectedResult, const std::string& name) {
  cali::Annotation mode("mode");
  mode.set(name.c_str());
  auto iteration = cali::Annotation("iteration");
  for (RAJA::Index_type i = 0; i < numTrials; ++i) {
    iteration.set(i);
    auto actualResult = f(x, L, M);
    iteration.set("test");
    checkResult(actualResult, expectedResult, L*M);
  }
  iteration.end();
  mode.end();
}

int main() {
    constexpr const std::array<int, 10> sizes{8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    constexpr const double sampleRate = 40;
    constexpr const double inputFrequency = 10;
    constexpr const int NUM_TRIALS = 10;

    auto size = cali::Annotation("size");

    for (auto numSamples : sizes) {
      numSamples = 1 << numSamples;
      size.set(numSamples);
      std::cout << "Starting size " << numSamples << "\n";

      std::vector<double> sine(numSamples);
      for (int i = 0; i < numSamples; ++i) {
        sine[i] = 20*sin(2*PI * inputFrequency * (i / sampleRate));
      }

      const int L = 4;
      const int M = 1 << 6;

      auto control = cali::Annotation("mode");
      control.set("control");
      std::complex<double>* controlBins;
      auto iteration = cali::Annotation("iteration");
      for (RAJA::Index_type i = 0; i < NUM_TRIALS; ++i) {
        iteration.set(i);
        controlBins = fft2dSequential(&sine[0], L, M);
      }
      iteration.end();
      control.end();
      std::cout << "Completed control without error.\n";

      runTimingTest(fft2dRajaSequential, &sine[0], L, M, NUM_TRIALS, controlBins, "Serial");
      std::cout << "Completed 2D FFT sequential style without error.\n";

      runTimingTest(fft2dRajaOmp, &sine[0], L, M, NUM_TRIALS, controlBins, "OpenMP");
      std::cout << "Completed 2D FFT omp style without error.\n";
    }
    size.end();
}
