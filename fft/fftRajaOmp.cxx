#include <math.h>
#include <complex>
#include <iostream>

#include "RAJA/RAJA.hxx"
#include "caliper/Annotation.h"

const double PI = acos(-1);

//This is a sequential implementation of a FFT using
void fftSeq(double* x, std::complex<double>* bins, int N, int s)
{

    //Base case if we have a DFT of size 1
    if (N == 1){
        bins[0] = std::complex<double> (x[0],0);

    } else {
        
        //Determin the fft of the two different halves
        if (s == 1){
            auto forTime = cali::Annotation("largeFor").begin();
            RAJA::forall<RAJA::seq_exec>(0, 2, [&](int j)
            {   
                fftSeq(x+j*s,bins + j*N/2, N/2, 2*s); 
            });
            forTime.end();

        }else{
            fftSeq(x,bins, N/2, 2*s);
            fftSeq(x+s,bins + N/2, N/2, 2*s); 
        }
        
         
        

        //Make the term I which is a complex double = sqrt -1
        std::complex<double> I = -1.0;
        I = sqrt(I) ;
        RAJA::forall<RAJA::seq_exec>(0, N/2, [&](int k)
        {   
            std::complex<double> a = (-2*k*PI)/N;
            std::complex<double> t = bins[k];
            bins[k] = t + exp(a*I)*bins[k + N/2];
            bins[k+N/2] = t - exp(a*I)*bins[k + N/2];
        });
    }
}

int main() {
    
    //The number of samples which were collected(must be a power of 2)
    int numberSamples = pow(2, 16);
    //The rate at which the samples were taken in Hz
    double sampleRate = 40.0;

    //make sure this is less than half of sample rate
    double inputFreq = 12.67;

    //make the input array
    double sine[numberSamples];

    std::complex<double>* bins = new std::complex<double>[numberSamples];

    for (int i = 0; i < numberSamples; ++i)
    {
        sine[i] = 20*sin(2*PI * inputFreq * (i / sampleRate));
    }

    auto totalTime = cali::Annotation("totalFFT").begin();
    //get the bins from the fft
    fftSeq(sine, bins, numberSamples, 1);
    totalTime.end();
    
    //Print out the results of the fft but adjust the numbers so they correspond 
    for (int i = 0; i < numberSamples; ++i){
        std::cout << (sampleRate/numberSamples)*i <<": " << std::abs(bins[i]) << std::endl;
    }

    return 0;
}