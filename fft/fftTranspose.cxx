#include <math.h>
#include <complex>
#include <iostream>

#include "RAJA/RAJA.hxx"

const double PI = acos(-1);

//This is a sequential implementation of a FFT using RAJA
std::complex<double>* fftSeq(double* x, int N, int s)
{
    std::complex<double>* bin = new std::complex<double>[N];

    //Base case if we have a DFT of size 1
    if (N == 1){
        bin[0] = std::complex<double> (x[0],0);

    } else {
        //Make a tempory bin for pulling out the arrays from the two halfs
        std::complex<double>* tempBin = fftSeq(x, N/2, 2*s);
        
        for (int i = 0; i < N/2; ++i ){
            bin[i] = tempBin[i];
        }

        //make sure to delete the dynamically allocated memory
        delete tempBin;

        tempBin = fftSeq(x+s, N/2, 2*s);

        for(int i = 0; i < N/2; ++i){
            bin[N/2 + i] = tempBin[i];

        }
        //make sure to delete the dynamically allocated memory
        delete tempBin;

        //Make the term I which is a complex double = sqrt -1
        std::complex<double> I = -1.0;
        I = sqrt(I) ;
        RAJA::forall<RAJA::seq_exec>(0, N/2, [&](int k)
        {   
            std::complex<double> a = (-2*k*PI)/N;
            std::complex<double> t = bin[k];
            bin[k] = t + exp(a*I)*bin[k + N/2];
            bin[k+N/2] = t - exp(a*I)*bin[k + N/2];
        });
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
    std::complex<double>* bins = fftSeq(sine, numberSamples, 1);
    
    //Print out the results of the fft but adjust the numbers so they correspond 
    for (int i = 0; i < numberSamples; ++i){
        std::cout << (sampleRate/numberSamples)*i <<": " << std::abs(bins[i]) << std::endl;
    }

    return 0;
}