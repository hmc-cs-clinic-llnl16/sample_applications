#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cassert>

#include <iostream>
#include <vector>
#include <type_traits>
#include <random>
#include <string>
#include <sstream>
#include <stdexcept>

#include "RAJA/RAJA.hxx"

#include "caliper/Annotation.h"

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
      RAJA::RangeSegment(0, cols_),
      [=](RAJA::Index_type row, RAJA::Index_type col, RAJA::Index_type k) {
        resultView(row, col) += rhsView(row, k) * lhsView(k, col);
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
    RAJA::ExecList<RAJA::simd_exec, 
                   RAJA::simd_exec,
                   RAJA::simd_exec>,
    RAJA::Permute<RAJA::PERM_IJK>
  >;
  using VIEW = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = "Serial";
};

struct OmpPolicy {
  using EXEC = RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::omp_parallel_for_exec, 
                   RAJA::simd_exec,
                   RAJA::simd_exec>,
    RAJA::Permute<RAJA::PERM_IJK>
  >;
  using VIEW = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = "OMP";
};

struct AgencyPolicy {
  using EXEC = RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::agency_parallel_exec, 
                   RAJA::simd_exec,
                   RAJA::simd_exec>,
    RAJA::Permute<RAJA::PERM_IJK>
  >;
  using VIEW = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = "Agency";
};

struct AgencyOMPPolicy {
  using EXEC = RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::agency_omp_parallel_exec, 
                   RAJA::simd_exec,
                   RAJA::simd_exec>,
    RAJA::Permute<RAJA::PERM_IJK>
  >;
  using VIEW = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = "AgencyOMP";
};

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

template <typename T>
std::vector<T> multOMP(const std::vector<T>& lhs,
                       const std::vector<T>& rhs,
                       const std::size_t rows,
                       const std::size_t cols) {
  std::vector<T> resultV(rows * cols, 0);

  #pragma omp parallel for
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

template <typename MATRIX>
void runTimingTest(MATRIX& left, MATRIX& right, const std::vector<double>& result, const unsigned numTrials) {
    cali::Annotation::Guard timing_test(cali::Annotation(MATRIX::name).begin());
    auto iteration = cali::Annotation("iteration");
    for (RAJA::Index_type i = 0; i < numTrials; ++i) {
      std::cout << "Started iteration " << i << " of type " << MATRIX::name << "\n";
      iteration.set(i);
      auto actualResult = left * right;
      iteration.set("test");
      checkResult(actualResult, result);
    }
    iteration.end();
}

template <typename MATRIX>
void checkResult(const MATRIX& actual, const std::vector<double>& expected) {
    for (std::size_t i = 0; i < actual.getRows(); ++i) {
      for (std::size_t j = 0; j < actual.getCols(); ++j) {
        if (actual(i, j) != expected[i * actual.getCols() + j]) {
          std::ostringstream os;
          os << "Invalid value at: (" << i << ", " << j << "). Was " << actual(i, j) << " but expected " << expected[i * actual.getCols()  + j] << "\n";
          throw std::runtime_error(os.str());
        }
      }
    }
}

int main(int argc, char** argv) {
  constexpr static std::size_t numSizes = 10;
  constexpr static std::size_t NUM_TRIALS = 10;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0, 5);

  auto size = cali::Annotation("size");
  for (int sizeIndex = 0; sizeIndex < numSizes; ++sizeIndex) {
    auto rows = interpolateNumberLinearlyOnLogScale(100, 1000, numSizes, sizeIndex);
    auto cols = rows;
    size.set(rows);
    std::cout << "Starting size " << rows << "\n";

    auto init = cali::Annotation("initialization").begin();
    auto matrix1 = Matrix<double, SerialPolicy>(rows, cols);
    auto matrix2 = Matrix<double, SerialPolicy>(rows, cols);
    auto matrix3 = Matrix<double, OmpPolicy>(rows, cols);
    auto matrix4 = Matrix<double, OmpPolicy>(rows, cols);
    auto matrix5 = Matrix<double, AgencyPolicy>(rows, cols);
    auto matrix6 = Matrix<double, AgencyPolicy>(rows, cols);
    auto matrix7 = Matrix<double, AgencyOMPPolicy>(rows, cols);
    auto matrix8 = Matrix<double, AgencyOMPPolicy>(rows, cols);

    auto control1 = std::vector<double>(rows * cols);
    auto control2 = std::vector<double>(rows * cols);

    auto rawOMP1 = std::vector<double>(rows * cols);
    auto rawOMP2 = std::vector<double>(rows * cols);

    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        double first = dist(gen), second = dist(gen);
        matrix1(i, j) = first;
        matrix3(i, j) = first;
        matrix5(i, j) = first;
        matrix7(i, j) = first;
        rawOMP1[i * cols + j] = first;
        control1[i * cols + j] = first;

        matrix2(i, j) = second;
        matrix4(i, j) = second;
        matrix6(i, j) = second;
        matrix8(i, j) = second;
        rawOMP2[i * cols + j] = second;
        control2[i * cols + j] = second;
      }
    }
    init.end();

    auto control = cali::Annotation("control").begin();
    std::vector<double> resultV;
    auto iteration = cali::Annotation("iteration");
    for (RAJA::Index_type i = 0; i < NUM_TRIALS; ++i) {
      std::cout << "Started iteration " << i << " of type control\n";
      iteration.set(i);
      resultV = mult(control1, control2, rows, cols);
    }
    iteration.end();
    control.end();

    auto omp = cali::Annotation("rawOMP").begin();
    std::vector<double> ompResult;
    auto ompiteration = cali::Annotation("iteration");
    for (RAJA::Index_type i = 0; i < NUM_TRIALS; ++i) {
      std::cout << "Started iteration " << i << "of type raw OpenMP\n";
      iteration.set(i);
      ompResult = multOMP(rawOMP1, rawOMP2, rows, cols);
    }
    ompiteration.end();
    omp.end();

    try {
      runTimingTest(matrix1, matrix2, resultV, NUM_TRIALS);
    } catch (std::runtime_error e) {
      std::cout << e.what() << std::endl;
      return 1;
    }

    std::cout << "Completed matrix multiplication serial style without error.\n";

    try {
      runTimingTest(matrix3, matrix4, resultV, NUM_TRIALS);
    } catch (std::runtime_error e) {
      std::cout << e.what() << std::endl;
      return 1;
    }

    std::cout << "Completed matrix multiplication omp style without error." << std::endl;

    try {
      runTimingTest(matrix5, matrix6, resultV, NUM_TRIALS);
    } catch (std::runtime_error e) {
      std::cout << e.what() << std::endl;
      return 1;
    }

    std::cout << "Completed matrix multiplication agency style without error." << std::endl;

    try {
      runTimingTest(matrix7, matrix8, resultV, NUM_TRIALS);
    } catch (std::runtime_error e) {
      std::cout << e.what() << std::endl;
      return 1;
    }

    std::cout << "Completed matrix multiplication agency_omp style without error." << std::endl;
  }
  size.end();
}
