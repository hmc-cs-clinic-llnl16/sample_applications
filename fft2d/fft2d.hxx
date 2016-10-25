#if !defined(SAMPLE_APPLICATIONS_2D_FFT)
#define SAMPLE_APPLICATIONS_2D_FFT

#include <math.h>
#include <complex>
#include <iostream>

#include "RAJA/RAJA.hxx"

extern const double PI;
extern const std::complex<double> A;

std::complex<double>* fft2dSequential(double* x, int L, int M);

std::complex<double>* fft2dRajaSequential(double* x, int L, int M);

std::complex<double>* fft2dRajaOmp(double* x, int L, int M);

#endif //defined SAMPLE_APPLICATIONS_2D_FFT
