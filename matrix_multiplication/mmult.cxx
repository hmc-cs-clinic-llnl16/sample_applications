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

template <typename MATRIX>
void runTimingText(MATRIX& left, MATRIX& right, const std::vector<double>& result, const unsigned numTrials) {
    cali::Annotation::Guard timing_test(cali::Annotation(MATRIX::name).begin());
    auto iteration = cali::Annotation("iteration");
    for (RAJA::Index_type i = 0; i < numTrials; ++i) {  
      std::cout << "Started iteration " << i << " of type " << MATRIX::name << "... "; 
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
  constexpr static RAJA::Index_type NUM_ROWS = 1000;
  constexpr static RAJA::Index_type NUM_COLS = 1000;
  constexpr static std::size_t NUM_TRIALS = 10;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0, 5);

  auto init = cali::Annotation("initialization").begin();
  auto matrix1 = Matrix<double, SerialPolicy>(NUM_ROWS, NUM_COLS);
  auto matrix2 = Matrix<double, SerialPolicy>(NUM_ROWS, NUM_COLS);
  auto matrix3 = Matrix<double, OmpPolicy>(NUM_ROWS, NUM_COLS);
  auto matrix4 = Matrix<double, OmpPolicy>(NUM_ROWS, NUM_COLS);

  auto control1 = std::vector<double>(NUM_ROWS * NUM_COLS);
  auto control2 = std::vector<double>(NUM_ROWS * NUM_COLS);

  for (std::size_t i = 0; i < NUM_ROWS; ++i) {
    for (std::size_t j = 0; j < NUM_COLS; ++j) {
      double first = dist(gen), second = dist(gen);
      matrix1(i, j) = first;
      matrix3(i, j) = first;
      control1[i * NUM_COLS + j] = first;
      matrix2(i, j) = second;
      matrix4(i, j) = second;
      control2[i * NUM_COLS + j] = second;
    }
  }
  init.end();

  auto control = cali::Annotation("control").begin();
  auto resultV = mult(control1, control2, NUM_ROWS, NUM_COLS);
  control.end();
  try {
    runTimingText(matrix1, matrix2, resultV, NUM_TRIALS);
  } catch (std::runtime_error e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  std::cout << "Completed matrix multiplication serial style without error.\n";

  try {
    runTimingText(matrix3, matrix4, resultV, NUM_TRIALS);
  } catch (std::runtime_error e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  std::cout << "Completed matrix multiplication omp style without error." << std::endl;
}

