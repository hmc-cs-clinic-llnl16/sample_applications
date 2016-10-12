#include <math.h>
#include <complex>
#include <iostream>

#include "fft2d.hxx"

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
