#include "caliper/Annotation.h"
#include <string>
#include <utility>
#include <iostream>

template <typename Functor, typename CheckFunctor, typename Result, typename... Args>
void runTimingTest(Functor& f, const Result& expectedResult, const std::string& description, const std::size_t numTrials, Args&&... args);

