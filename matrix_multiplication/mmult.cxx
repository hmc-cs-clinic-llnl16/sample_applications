#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cassert>

#include <iostream>
#include <functional>
#include <vector>
#include <type_traits>
#include <random>
#include <string>
#include <sstream>
#include <stdexcept>

#include "RAJA/RAJA.hxx"

#include "caliper/Annotation.h"

#include "../common/timing.hxx"

template <typename T, typename Policy>
class Matrix {
  RAJA::Index_type rows_;
  RAJA::Index_type cols_;
  std::vector<T> data_;
public:
  static constexpr char* name = Policy::name;
  Matrix() { }
  
  Matrix(const RAJA::Index_type rows, const RAJA::Index_type cols, const T& defaultValue = T())
      : rows_(rows), cols_(cols), data_(rows_ * cols_, defaultValue) { }

  Matrix<T, Policy> operator*(Matrix<T, Policy>& lhs) {
    assert(lhs.cols_ == rows_ && lhs.rows_ == cols_);

    auto result = Matrix<T, Policy>(rows_, cols_, T());

    typename Policy::VIEW rhsView(&data_[0], rows_, cols_);
    typename Policy::VIEW lhsView(&lhs.data_[0], rows_, cols_);
    typename Policy::VIEW resultView(&result.data_[0], rows_, cols_);

    RAJA::forallN<typename Policy::EXEC>(
      RAJA::RangeSegment(0, rows_),
      RAJA::RangeSegment(0, cols_),
      [=](RAJA::Index_type row, RAJA::Index_type col) {
        double result = 0;
        for (RAJA::Index_type k = 0; k < rows_; ++k) {
          result += rhsView(row, k) * lhsView(k, col);
        }
        resultView(row, col) = result;
      }
    );

    return result;
  }

  RAJA::Index_type getRows() const { return rows_; }
  RAJA::Index_type getCols() const { return cols_; }

  T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) {
    return data_[row * cols_ + col];
  }

  const T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) const {
    return data_[row * cols_ + col];
  }
};

struct SerialPolicy {
  using EXEC = RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>,
    RAJA::Permute<RAJA::PERM_IJ>
  >;
  using VIEW = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = "Serial";
};

struct OmpPolicy {
  using EXEC = RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::seq_exec>,
    RAJA::Permute<RAJA::PERM_IJ>
  >;
  using VIEW = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = "OMP";
};

template <typename M>
M mmult(M& lhs, M& rhs) {
  return lhs * rhs;
}

template <typename T>
std::vector<T> mult(const std::vector<T>& lhs, const std::vector<T>& rhs, const std::size_t rows, const std::size_t cols) {
  std::vector<T> resultV(rows * cols, 0);

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      double result = 0;
      for (std::size_t k = 0; k < rows; ++k) {
        result += lhs[row * cols + k] * rhs[k * cols + col];
      }
      resultV[row * cols + col] = result;
    }
  }

  return resultV;
}

int interpolateNumberLinearlyOnLogScale(
    const int lower, const int upper,
    const unsigned int numberOfPoints,
    const unsigned int pointIndex) {
  const double percent =
    pointIndex / static_cast<double>(numberOfPoints - 1);
  const double power = std::log10(lower) +
    percent * (std::log10(upper) - std::log10(lower));
  return std::pow(10., power);
}

struct ResultCheckFunctor {
  std::vector<double> expectedResult;

  template <typename MATRIX>
  void operator()(const MATRIX& actual) {
    for (std::size_t i = 0; i < actual.getRows(); ++i) {
      for (std::size_t j = 0; j < actual.getCols(); ++j) {
        auto index = i * actual.getCols() + j;
        if (actual(i, j) != expectedResult[index]) {
          std::ostringstream os;
          os << "Invalid value at: (" << i << ", " << j << "). Was " << actual(i, j) << " but expected " << expectedResult[index] << "\n";
          throw std::runtime_error(os.str());
        }
      }
    }
  }
};

int main(int argc, char** argv) { 
  constexpr static std::size_t numSizes = 10;
  constexpr static std::size_t NUM_TRIALS = 10;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0, 5);

  auto size = cali::Annotation("size");
  for (int sizeIndex = 0; sizeIndex < numSizes; ++sizeIndex) {
    auto rows = interpolateNumberLinearlyOnLogScale(100, 5000, numSizes, sizeIndex);
    auto cols = rows;
    size.set(rows);
    std::cout << "Starting size " << rows << "\n";

    auto init = cali::Annotation("initialization").begin();
    auto matrix1 = Matrix<double, SerialPolicy>(rows, cols);
    auto matrix2 = Matrix<double, SerialPolicy>(rows, cols);
    auto matrix3 = Matrix<double, OmpPolicy>(rows, cols);
    auto matrix4 = Matrix<double, OmpPolicy>(rows, cols);

    auto control1 = std::vector<double>(rows * cols);
    auto control2 = std::vector<double>(rows * cols);

    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        double first = dist(gen), second = dist(gen);
        matrix1(i, j) = first;
        matrix3(i, j) = first;
        control1[i * cols + j] = first;
        matrix2(i, j) = second;
        matrix4(i, j) = second;
        control2[i * cols + j] = second;
      }
    }
    init.end();

    auto control = cali::Annotation("control").begin();
    auto resultV = mult(control1, control2, rows, cols);
    control.end();
    try {
      ResultCheckFunctor f{resultV};
      runTimingTest(mmult<Matrix<double, SerialPolicy>>, f, "RAJA Serial", NUM_TRIALS, matrix1, matrix2);
    } catch (std::runtime_error e) {
      std::cout << e.what() << std::endl;
      return 1;
    }

    std::cout << "Completed matrix multiplication serial style without error.\n";

    try {
      ResultCheckFunctor f{resultV};
      runTimingTest(mmult<Matrix<double, OmpPolicy>>, f, "RAJA OMP", NUM_TRIALS, matrix3, matrix4);
    } catch (std::runtime_error e) {
      std::cout << e.what() << std::endl;
      return 1;
    }

    std::cout << "Completed matrix multiplication omp style without error." << std::endl;
  }
  size.end();
}

