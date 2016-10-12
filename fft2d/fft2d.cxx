#include <math.h>

#include <array>
#include <complex>
#include <iostream>

#include "RAJA/RAJA.hxx"
#include "caliper/Annotation.h"

#include "fft2d.hxx"

const double PI = acos(-1);
const std::complex<double> A = -2*PI*1i;

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
  cali::Annotation::Guard timingTest(cali::Annotation(name.c_str()).begin());
  auto iteration = cali::Annotation("iteration");
  for (RAJA::Index_type i = 0; i < numTrials; ++i) {
    std::cout << "Started iteration " << i << " of type " << name << "\n";
    iteration.set(i);
    auto actualResult = f(x, L, M);
    iteration.set("test");
    checkResult(actualResult, expectedResult, L*M);
  }
  iteration.end();
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

      auto init = cali::Annotation("initialization").begin();
      std::vector<double> sine(numSamples);
      for (int i = 0; i < numSamples; ++i) {
        sine[i] = 20*sin(2*PI * inputFrequency * (i / sampleRate));
      }
      init.end();

      const int L = 4;
      const int M = 1 << 6;

      auto control = cali::Annotation("control").begin();
      std::complex<double>* controlBins;
      auto iteration = cali::Annotation("iteration");
      for (RAJA::Index_type i = 0; i < NUM_TRIALS; ++i) {
        std::cout << "Started control iteration " << i << "\n";
        iteration.set(i);
        controlBins = fft2dSequential(&sine[0], L, M);
      }
      iteration.end();
      control.end();

      try {
        runTimingTest(fft2dRajaSequential, &sine[0], L, M, NUM_TRIALS, controlBins, "Serial");
      } catch (std::runtime_error e) {
        std::cout << e.what() << std::endl;
        return 1;
      }

      std::cout << "Completed 2D FFT sequential style without error.\n";

      try {
        runTimingTest(fft2dRajaOmp, &sine[0], L, M, NUM_TRIALS, controlBins, "OMP");
      } catch (std::runtime_error e) {
        std::cout << e.what() << std::endl;
        return 1;
      }

      std::cout << "Completed 2D FFT omp style without error.\n";
    }
    size.end();
}