#include <iostream>
#include <math.h>

#include "RAJA/RAJA.hxx"
#include "RAJA/MemUtils_CPU.hxx"
#include "caliper/Annotation.h"

const int maxNum =  200000000;

double baseline(double* numberArray, int currNum){
	double sum = 0;
	for(int i = 0; i < currNum; ++i){
			sum += numberArray[i];
	}
	return sum;

}
double rajaSerial(double* numberArray, int currNum){
	RAJA::ReduceSum<RAJA::seq_reduce, double> reducer(0.0);

	RAJA::forall<RAJA::seq_exec>(0, currNum, [=](int i) {		
		reducer += numberArray[i];
	});

	return (double) reducer.get();
}

double ompParallel(double* numberArray, int currNum){
	RAJA::ReduceSum<RAJA::omp_reduce, double> reducer(0.0);

	RAJA::forall<RAJA::omp_parallel_for_exec>(0, currNum, [=](int i) {		
		reducer += numberArray[i];
	});

	return (double) reducer.get();
}

double agencyParallel(double* numberArray, int currNum){
	RAJA::ReduceSum<RAJA::agency_reduce, double> reducer(0.0);

	RAJA::forall<RAJA::agency_parallel_exec>(0, currNum, [=](int i) {		
		reducer += numberArray[i];
	});

	return (double) reducer.get();
}

double agencySerial(double* numberArray, int currNum){
	RAJA::ReduceSum<RAJA::agency_reduce, double> reducer(0.0);

	RAJA::forall<RAJA::agency_sequential_exec>(0, currNum, [=](int i) {		
		reducer += numberArray[i];
	});

	return (double) reducer.get();
}

void checkResult(double truth, double test, const std::string& name){
	if(std::fabs(truth - test) > 0.1){
		std::cout << "Wrong value encountered when reduceing using "  << name << std::endl;
		std::cout << "Expected "  << truth << " recived " << test << std::endl;
		//throw std::runtime_error("check result error");
	}
}

template <typename Functor>
void runTimingTest(Functor f, double* numberArray , int currNum, int numTrials, double expectedResult, const std::string& name) {
  cali::Annotation::Guard timingTest(cali::Annotation(name.c_str()).begin());
  auto iteration = cali::Annotation("iteration");
  for (RAJA::Index_type i = 0; i < numTrials; ++i) {
    std::cout << "Started iteration " << i << " of type " << name << "\n";
    iteration.set(i);
    auto actualResult = f(numberArray, currNum);
    iteration.set("test");
    checkResult(actualResult, expectedResult, name.c_str() );
  }
  iteration.end();
}

int main(){
	int testSet[] = {maxNum/100, maxNum/50, maxNum/10, maxNum/8, maxNum/4, maxNum/2, maxNum};
	int numTrials = 5;

	auto size = cali::Annotation("size");
	//For all of the different sizes
	for (int i = 0; i < 5; ++i){
		int currNum = testSet[i];
		double* numberArray = new double[currNum];
		double answer = (currNum * (currNum - 1.0)) / 2.0;

		size.set(currNum);

		double sum = 0.0;
		//initalize array
		for(int i = 0; i < currNum; ++i){
			numberArray[i] = i;
		}

		//Base Line
		runTimingTest(baseline, numberArray, currNum, numTrials, answer, "baseline");

		//RAJA Serial
		runTimingTest(rajaSerial, numberArray, currNum, numTrials, answer, "RajaSerial");

		//OMP Parallel
		runTimingTest(ompParallel, numberArray, currNum, numTrials, answer, "OMP");

		//Agency Parallel
		runTimingTest(agencyParallel, numberArray, currNum, numTrials, answer, "AgencyParallel");

		//Agency Serial
		runTimingTest(agencySerial, numberArray, currNum, numTrials, answer, "AgencySerial");
	}
	size.end();

	return 1;

	


	
}
