#include <math.h>
#include <complex>
#include <iostream>

const double PI = acos(-1);
const std::complex<double> A = -2*PI*1i;

std::complex<double>* fft2d(double* x, int L, int M) {

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
