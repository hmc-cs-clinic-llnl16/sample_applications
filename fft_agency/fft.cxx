#include <math.h>
#include <complex>
#include <iostream>

#include <agency/agency.hpp>

#include "caliper/Annotation.h"


const double PI = acos(-1);
//Make the term I which is a complex double = sqrt -1
const std::complex<double> I = sqrt(std::complex<double>(-1.0));


std::complex<double>* fftseq(double* x, int N, int s)
{
    std::complex<double>* bin = new std::complex<double>[N];

    //Base case if we have a DFT of size 1
    if (N == 1){
        bin[0] = std::complex<double> (x[0],0);

    } else {
        //Make a tempory bin for pulling out the arrays from the two halfs
        std::complex<double>* tempBin = fftseq(x, N/2, 2*s);
        
        for (int i = 0; i < N/2; ++i ){
            bin[i] = tempBin[i];
        }

        //make sure to delete the dynamically allocated memory
        delete[] tempBin;

        tempBin = fftseq(x+s, N/2, 2*s);

        for(int i = 0; i < N/2; ++i){
            bin[N/2 + i] = tempBin[i];

        }
        //make sure to delete the dynamically allocated memory
        delete[] tempBin;

        for (int k = 0; k < N / 2; ++k) {
            std::complex<double> a = (-2* k * PI)/N;
            std::complex<double> t = bin[k];
            bin[k] = t + exp(a * I) * bin[k + N / 2];
            bin[k+N/2] = t - exp(a*I)*bin[k + N/2];
        };
    }

    return bin;

}




template <typename Policy>
std::complex<double>* fft(double* x, int N, int s)
{
    std::complex<double>* bin = new std::complex<double>[N];

    //Base case if we have a DFT of size 1
    if (N == 1){
        bin[0] = std::complex<double> (x[0],0);

    } else {
        //Make a tempory bin for pulling out the arrays from the two halfs
        std::complex<double>* tempBin = fft<Policy>(x, N/2, 2*s);
        
        for (int i = 0; i < N/2; ++i ){
            bin[i] = tempBin[i];
        }

        //make sure to delete the dynamically allocated memory
        delete[] tempBin;

        tempBin = fft<Policy>(x+s, N/2, 2*s);

        for(int i = 0; i < N/2; ++i){
            bin[N/2 + i] = tempBin[i];

        }
        //make sure to delete the dynamically allocated memory
        delete[] tempBin;

        Policy policy;
        agency::bulk_invoke(policy(N / 2), [=](typename Policy::execution_agent_type& self)
        {
            int k = self.index();
            std::complex<double> a = (-2* k * PI)/N;
            std::complex<double> t = bin[k];
            bin[k] = t + exp(a * I) * bin[k + N / 2];
            bin[k+N/2] = t - exp(a*I)*bin[k + N/2];
        });
    }

    return bin;

}

template <typename Policy>
void runTimingTest(double* x, int N, int s, const unsigned numTrials, std::complex<double>* expectedResult)
{
    cali::Annotation::Guard timing_test(cali::Annotation(Policy::name).begin());
    auto iteration = cali::Annotation("iteration");
    for (int i = 0; i < numTrials; ++i) {
        std::cout << "Started iteration " << i << " of type " << Policy::name << "\n"; 
        iteration.set(i);
        auto result = fft<typename Policy::EXEC>(x, N, s);
        iteration.set("test");
    }
    iteration.end();
}

struct SequentialPolicy {
    using EXEC = agency::sequenced_execution_policy;
    using AGENT = agency::sequenced_agent;
    static constexpr char* name = "Sequential";
};

struct ParallelPolicy {
    using EXEC = agency::parallel_execution_policy;
    using AGENT = agency::parallel_agent;
    static constexpr char* name = "Parallel";
};


int main() {
    
    size_t startSamples = pow(2,8);
    size_t endSamples = pow(2, 12);
    int numTrials = 10;

    auto size = cali::Annotation("size");

    for (int numberSamples = startSamples; numberSamples <= endSamples; numberSamples *= 2) {
        //The number of samples which were collected(must be a power of 2)
        size.set(numberSamples);
        std::cout << "Starting size " << numberSamples << "\n";

        //The rate at which the samples were taken in Hz
        double sampleRate = 40.0;

        //make sure this is less than half of sample rate
        double inputFreq = 10.0;

        //make the input array

        auto init = cali::Annotation("initialization").begin();
        double sine[numberSamples];
        for (int i = 0; i < numberSamples; ++i) {
            sine[i] = 20*sin(2*PI * inputFreq * (i / sampleRate));
        }
        init.end();

        std::complex<double>* controlResult;
        auto control = cali::Annotation("control").begin();
        auto iteration = cali::Annotation("iteration");
        for (int i = 0; i < numTrials; ++i) {
            std::cout << "Started iteration " << i << " of type control\n";
            iteration.set(i);
            controlResult = fftseq(sine, numberSamples, 1);
            iteration.set("Free memory");
            if (i != numTrials - 1) {
                delete[] controlResult;
            }
        }
        control.end();
        //get the bins from the fft
        runTimingTest<SequentialPolicy>(sine, numberSamples, 1, numTrials, controlResult);
        runTimingTest<ParallelPolicy>(sine, numberSamples, 1, numTrials, controlResult);
    }
    size.end();

    return 0;
}

