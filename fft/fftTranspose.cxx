#include <math.h>
#include <complex>
#include <iostream>

#include "RAJA/RAJA.hxx"
#include "caliper/Annotation.h"

const double PI = acos(-1);



//This is a sequential implementation of an FFT 
void fft(std::complex<double>* x, std::complex<double>* bins, int N, int s)
{

    //Base case if we have a DFT of size 1
    if (N == 1){
        bins[0] = x[0];

    } else {
        
        //Determin the fft of the two different halves
        if (s == 1){
            //auto forTime = cali::Annotation("largeFor").begin();
            RAJA::forall<RAJA::seq_exec>(0, 2, [&](int j)
            {   
                fft(x+j*s,bins + j*N/2, N/2, 2*s); 
            });
            //forTime.end();

        }else{
            fft(x,bins, N/2, 2*s);
            fft(x+s,bins + N/2, N/2, 2*s); 
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

//This function takes in a single array in column major format and calcuates
//the fft of this sequence
template <typename Policy>
void cfft(std::complex<double>* x, std::complex<double>* bins, int n)
{   
    
    //Use pointer arithmatic to offset which part of the array to work on
    RAJA::forall<Policy>(0, n, [&](int k)
        {   

            fft(x + (n*k), bins + (n*k),n, 1);
        });
}

//This function transposes the 1D array assumeing it represents an nxn matrix
template <typename Policy>
void transpose(std::complex<double>* x, std::complex<double>* xTrans,int n)
{
    //Use raja so we can run multiple outter loops at the same time
    RAJA::forall<Policy>(0, n, [&](int k)
        {   
            for(int m = 0; m < n; ++m){
                xTrans[n*k + m] = x[n*m + k];
            }
        });
    
}
//This function multiply the imput by its corresponding twiddle factors
template <typename Policy>
void twiddle(std::complex<double>* x, std::complex<double>* xTwid, std::complex<double>* factors, int n)
{
    //Multiple every term by the twiddle factor   
    RAJA::forall<Policy>(0, n*n, [&](int k)
        {
            xTwid[k] = x[k]*factors[k];
        }); 
}

//This function generates the two dimensional twiddle factors in a nxn
//matrix
void genTwid(int n, std::complex<double>* factors)
{
    //Get the omega n factor
    std::complex<double> wn = std::complex<double> (cos((2.0*PI)/(n*n)), -sin((2.0*PI)/(n*n)));

    RAJA::forall<RAJA::omp_parallel_for_exec>(0, n*n, [&](int k)
      {
          factors[k] = pow(wn, (k%n) * (k/n));
      });
}

//This is an implementation of a parallel version of fft
template <typename Policy>
void sixStepFFT(std::complex<double>* x, std::complex<double>* bins, int numberSamples, 
    int n, std::complex<double>* factors)
{   

    //Transpose the matrix and put the result into t1
    std::complex<double>* t1 = new std::complex<double>[numberSamples];
    transpose<Policy>(x,t1,n);


    //Calculate the collumn fft and put it into c1
    std::complex<double>* c1 = new std::complex<double>[numberSamples];
    cfft<Policy>(t1, c1, n);

    delete t1;



    //Multiply the array by the twiddle factors
    std::complex<double>* twid = new std::complex<double>[numberSamples];
    twiddle<Policy>(c1, twid, factors, n);

    delete c1;

    //Transpose the matrix again
    std::complex<double>* t2 = new std::complex<double>[numberSamples];
    transpose<Policy>(twid, t2,n);

    delete twid;

    //Calculate the new collumn fft and put it into c1
    std::complex<double>* c2 = new std::complex<double>[numberSamples];
    cfft<Policy>(t2, c2,n);

    delete t2;
    //Now we have the last transpose which will give us the final results
    transpose<Policy>(c2,bins,n); 

    delete c2;

}

void fftSeq(std::complex<double>* x, std::complex<double>* bins, int N, int s)
{

    //Base case if we have a DFT of size 1
    if (N == 1){
        bins[0] = x[0];
    } else {
        
        //Determin the fft of the two different halves
        fftSeq(x,bins, N/2, 2*s);
        
        fftSeq(x+s,bins + N/2, N/2, 2*s);

        //Make the term I which is a complex double = sqrt -1
        std::complex<double> I = -1.0;
        I = sqrt(I) ;
        for (int k = 0; k < N/2; ++k)
        {   
            std::complex<double> a = (-2*k*PI)/N;
            std::complex<double> t = bins[k];
            bins[k] = t + exp(a*I)*bins[k + N/2];
            bins[k+N/2] = t - exp(a*I)*bins[k + N/2];
        }
    }

}

int main() {

    //The rate at which the samples were taken in Hz
    double sampleRate = 40.0;

    //make sure this is less than half of sample rate
    double inputFreq = 10.2;

    const int NUM_TRIALS = 10;

    auto sizeAnn = cali::Annotation("size");
    for (int size = 16; size < 24; ++size){// += 2) {
        int numberSamples = pow(2, size);
        int n = pow(2, size/2);
        sizeAnn.set(numberSamples);
        std::cout << "Starting size " << size << std::endl;

        std::complex<double>* x = new std::complex<double>[numberSamples];
        std::complex<double>* bins = new std::complex<double>[numberSamples];

        for (int i = 0; i < numberSamples; ++i)
        {
            x[i] = std::complex<double> (3.0*sin(2.0*PI * inputFreq * (i / sampleRate)), 0.0);
        }

        //Generate the twiddle factors
        std::complex<double>* factors = new std::complex<double>[numberSamples];
        genTwid(n,factors);

        auto iteration = cali::Annotation("iteration");
        for (int trialId = 0; trialId < NUM_TRIALS; ++trialId) {
            std::cout << "Beginning iteration " << trialId << std::endl;
            iteration.set(trialId);

            std::cout << "control\n";
            auto control = cali::Annotation("mode");
            control.set("control");
            fftSeq(x, bins, numberSamples, 1);
            control.end();

            std::cout << "Serial\n";
            auto sequential = cali::Annotation("mode");
            sequential.set("Serial");
            fft(x,bins, numberSamples,1);
            sequential.end();

            std::cout << "Omp\n";
            auto omp = cali::Annotation("mode");
            omp.set("OpenMP");
            sixStepFFT<RAJA::omp_parallel_for_exec>(x, bins, numberSamples, n, factors);
            omp.end();

            std::cout << "Agency\n";
            auto agency = cali::Annotation("mode");
            agency.set("Agency");
            sixStepFFT<RAJA::agency_parallel_exec>(x, bins, numberSamples, n, factors);
            agency.end();

            std::cout <<"aomp\n";
            auto agencyomp = cali::Annotation("mode");
            agencyomp.set("AgencyOmp");
            sixStepFFT<RAJA::agency_omp_parallel_exec>(x, bins, numberSamples, n, factors);
            agencyomp.end();
        }
        iteration.end();

        delete x;
        delete bins;
    }
    sizeAnn.end();
}
