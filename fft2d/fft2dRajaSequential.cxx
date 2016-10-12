#include <math.h>
#include <complex>
#include <iostream>

#include "RAJA/RAJA.hxx"

#include "fft2d.hxx"

typedef RAJA::seq_exec exec_policy;
typedef RAJA::seq_reduce reduce_policy;

std::complex<double>* fft2dRajaSequential(double* x, int L, int M) {

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
