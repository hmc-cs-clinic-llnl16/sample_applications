#include "caliper/Annotation.h"

#include "timing.hxx"

#include <iostream>
#include <string>
#include <utility>

template <typename Functor, typename CheckFunctor, typename... Args>
void runTimingTest(Functor& f, CheckFunctor& cf, const std::string& description, const std::size_t numTrials, Args&&... args) {
  cali::Annotation::Guard timing_test(cali::Annotation(description).begin());
  auto iteration = cali::Annotation("iteration");
  for (int i = 0; i < numTrials; ++i) {
    std::cout << "Started iteration " << i+1 << " of " << numTrials << " for " << description << "\n";
    iteration.set(i);
    auto result = f(std::forward<Args>(args)...);
    iteration.set("test");
    cf(result);
  }
  iteration.end();
}
