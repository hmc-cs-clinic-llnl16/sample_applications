#include <ctime>
#include <cmath>
#include <cstring>
#include <string.h>
#include <cstdlib>
#include <cassert>

#include <iostream>
#include <vector>
#include <type_traits>
#include <random>
#include <sstream>
#include <stdexcept>

#include "RAJA/RAJA.hxx"
#include "view.hxx"

template <typename Policy>
class Matrix {
  Data<typename Policy::MemSpace> data_;

  RAJA::Index_type rows_;
  RAJA::Index_type cols_;
public:
  Matrix() { }

  Matrix(const RAJA::Index_type rows, const RAJA::Index_type cols, const RAJA::Real_type& defaultValue = 0)
      : rows_(rows), cols_(cols), data_(rows_ * cols_, defaultValue) { }

  Matrix<Policy> operator*(Matrix<Policy>& lhs) {
    assert(lhs.cols_ == rows_ && lhs.rows_ == cols_);

    auto result = Matrix<Policy>(rows_, cols_, 0);

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

  RAJA::Real_type& operator()(const RAJA::Index_type row, const RAJA::Index_type col) {
    return data_[row * cols_ + col];
  }

  const RAJA::Real_type& operator()(const RAJA::Index_type row, const RAJA::Index_type col) const {
    return data_[row * cols_ + col];
  }
};

template <bool Gpu>
struct MemorySpace {
  constexpr static bool IsGPU = Gpu;
};

struct SerialPolicy {
  using EXEC = RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>,
    RAJA::Permute<RAJA::PERM_IJ>
  >;
  using VIEW = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  using MemSpace = MemorySpace<false>;
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

int main(int argc, char** argv) {
  constexpr static RAJA::Index_type NUM_ROWS = 10;
  constexpr static RAJA::Index_type NUM_COLS = 10;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0, 5);

  auto matrix1 = Matrix<SerialPolicy>(NUM_ROWS, NUM_COLS);
  auto matrix2 = Matrix<SerialPolicy>(NUM_ROWS, NUM_COLS);

  auto control1 = std::vector<double>(NUM_ROWS * NUM_COLS);
  auto control2 = std::vector<double>(NUM_ROWS * NUM_COLS);

  for (std::size_t i = 0; i < NUM_ROWS; ++i) {
    for (std::size_t j = 0; j < NUM_COLS; ++j) {
      double first = dist(gen), second = dist(gen);
      matrix1(i, j) = first;
      control1[i * NUM_COLS + j] = first;
      matrix2(i, j) = second;
      control2[i * NUM_COLS + j] = second;
    }
  }

  auto result = matrix1 * matrix2;
  auto resultV = mult(control1, control2, NUM_ROWS, NUM_COLS);

  for (std::size_t i = 0; i < NUM_ROWS; ++i) {
    for (std::size_t j = 0; j < NUM_COLS; ++j) {
      if (result(i, j) != resultV[i * NUM_COLS + j]) {
        std::cout << "Invalid value at: (" << i << ", " << j << "). Was " << result(i, j) << " but expected " << resultV[i * NUM_COLS + j] << std::endl;
        return 1;
      }
    }
  }
}

