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
class Matrix3D {
  RAJA::Index_type rows_;
  RAJA::Index_type cols_;
  std::vector<T> data_;
public:
  using View = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = Policy::name;
  Matrix3D() { }

  Matrix3D(const RAJA::Index_type rows, const RAJA::Index_type cols, const T& defaultValue = T())
      : rows_(rows), cols_(cols), data_(rows_ * cols_, defaultValue) { }

  RAJA::Index_type getRows() const { return rows_; }
  RAJA::Index_type getCols() const { return cols_; }

  Matrix3D<T, Policy> operator*(Matrix3D<T, Policy>& rhs) {
    assert(rhs.cols_ == rows_ && rhs.rows_ == cols_);

    auto result = Matrix3D<T, Policy>(rows_, cols_, T());

    View lhsView(&data_[0], rows_, cols_);
    View rhsView(&rhs.data_[0], rows_, cols_);
    View resultView(&result.data_[0], rows_, cols_);

    RAJA::forallN<typename Policy::EXEC>(
      RAJA::RangeSegment(0, rows_),
      RAJA::RangeSegment(0, cols_),
      RAJA::RangeSegment(0, cols_),
      [=](RAJA::Index_type row, RAJA::Index_type col, RAJA::Index_type k) {
          resultView(row, col) += lhsView(row, k) * rhsView(k, col);
      }
    );

    return result;
  }

  T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) {
    return data_[row * cols_ + col];
  }

  const T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) const {
    return data_[row * cols_ + col];
  }
};

struct SerialPolicy3D {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec, RAJA::simd_exec>,
      RAJA::Permute<RAJA::PERM_IJK>>;
  static constexpr char* name = "Serial";
};

struct OmpPolicy3D {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec, RAJA::simd_exec>,
      RAJA::Permute<RAJA::PERM_IJK>>;
  static constexpr char* name = "OMP";
};

struct AgencyPolicy3D {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::agency_parallel_exec, RAJA::simd_exec, RAJA::simd_exec>,
      RAJA::Permute<RAJA::PERM_IJK>>;
  static constexpr char* name = "Agency";
};

struct AgencyOMPPolicy3D {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::agency_omp_parallel_exec, RAJA::simd_exec, RAJA::simd_exec>,
      RAJA::Permute<RAJA::PERM_IJK>>;
  static constexpr char* name = "AgencyOMP";
};

struct SerialPolicy3D_IKJ {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec, RAJA::simd_exec>,
      RAJA::Permute<RAJA::PERM_IKJ>>;
  static constexpr char* name = "Serial";
};

struct OmpPolicy3D_IKJ {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec, RAJA::simd_exec>,
      RAJA::Permute<RAJA::PERM_IKJ>>;
  static constexpr char* name = "OMP";
};

struct AgencyPolicy3D_IKJ {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::agency_parallel_exec, RAJA::simd_exec, RAJA::simd_exec>,
      RAJA::Agency_Parallel<RAJA::agency_parallel_exec::Agent_t,
                            RAJA::agency_parallel_exec::Worker_t,
                            RAJA::Permute<RAJA::PERM_IKJ>>>;
  static constexpr char* name = "Agency";
};

struct AgencyOMPPolicy3D_IKJ {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::agency_omp_parallel_exec, RAJA::simd_exec, RAJA::simd_exec>,
      RAJA::Agency_Parallel<RAJA::agency_omp_parallel_exec::Agent_t,
                            RAJA::agency_omp_parallel_exec::Worker_t,
                            RAJA::Permute<RAJA::PERM_IKJ>>>;
  static constexpr char* name = "AgencyOMP";
};

template <typename T, typename Policy>
class Matrix2D_IJK {
  RAJA::Index_type rows_;
  RAJA::Index_type cols_;
  std::vector<T> data_;
public:
  using View = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = Policy::name;
  Matrix2D_IJK() { }

  Matrix2D_IJK(const RAJA::Index_type rows, const RAJA::Index_type cols, const T& defaultValue = T())
      : rows_(rows), cols_(cols), data_(rows_ * cols_, defaultValue) { }

  RAJA::Index_type getRows() const { return rows_; }
  RAJA::Index_type getCols() const { return cols_; }

  Matrix2D_IJK<T, Policy> operator*(Matrix2D_IJK<T, Policy>& rhs) {
    assert(rhs.cols_ == rows_ && rhs.rows_ == cols_);

    auto result = Matrix2D_IJK(rows_, cols_, T());

    View lhsView(&data_[0], rows_, cols_);
    View rhsView(&rhs.data_[0], rows_, cols_);
    View resultView(&result.data_[0], rows_, cols_);

    RAJA::forallN<typename Policy::EXEC>(
      RAJA::RangeSegment(0, rows_),
      RAJA::RangeSegment(0, cols_),
      [=](RAJA::Index_type row, RAJA::Index_type col) {
          T tmp = 0;
          for (RAJA::Index_type k = 0; k < cols_; ++k) {
              tmp += lhsView(row, k) * rhsView(k, col);
          }
          resultView(row, col) = tmp;
      }
    );

    return result;
  }


  T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) {
    return data_[row * cols_ + col];
  }

  const T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) const {
    return data_[row * cols_ + col];
  }
};

template <typename T, typename Policy>
class Matrix2D_IKJ {
  RAJA::Index_type rows_;
  RAJA::Index_type cols_;
  std::vector<T> data_;
public:
  using View = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = Policy::name;
  Matrix2D_IKJ() { }

  Matrix2D_IKJ(const RAJA::Index_type rows, const RAJA::Index_type cols, const T& defaultValue = T())
      : rows_(rows), cols_(cols), data_(rows_ * cols_, defaultValue) { }

  RAJA::Index_type getRows() const { return rows_; }
  RAJA::Index_type getCols() const { return cols_; }

  Matrix2D_IKJ<T, Policy> operator*(Matrix2D_IKJ<T, Policy>& rhs) {
    assert(rhs.cols_ == rows_ && rhs.rows_ == cols_);

    auto result = Matrix2D_IKJ(rows_, cols_, T());

    View lhsView(&data_[0], rows_, cols_);
    View rhsView(&rhs.data_[0], rows_, cols_);
    View resultView(&result.data_[0], rows_, cols_);

    RAJA::forallN<typename Policy::EXEC>(
      RAJA::RangeSegment(0, rows_),
      RAJA::RangeSegment(0, cols_),
      [=](RAJA::Index_type row, RAJA::Index_type k) {
          for (RAJA::Index_type col = 0; col < cols_; ++col) {
              resultView(row, col) += lhsView(row, k) * rhsView(k, col);
          }
      }
    );

    return result;
  }

  T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) {
    return data_[row * cols_ + col];
  }

  const T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) const {
    return data_[row * cols_ + col];
  }
};

struct SerialPolicy2D {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::simd_exec, RAJA::simd_exec>,
      RAJA::Permute<RAJA::PERM_IJ>>;
  static constexpr char* name = "Serial";
};

struct OmpPolicy2D {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::simd_exec>,
      RAJA::Permute<RAJA::PERM_IJ>>;
  static constexpr char* name = "OMP";
};

struct AgencyPolicy2D {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::agency_parallel_exec, RAJA::simd_exec>,
      RAJA::Permute<RAJA::PERM_IJ>>;
  static constexpr char* name = "Agency";
};

struct AgencyOMPPolicy2D {
  using EXEC = RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::agency_omp_parallel_exec, RAJA::simd_exec>,
      RAJA::Permute<RAJA::PERM_IJ>>;
  static constexpr char* name = "AgencyOMP";
};

template <typename T, typename Policy>
class Matrix1D_IJK {
  RAJA::Index_type rows_;
  RAJA::Index_type cols_;
  std::vector<T> data_;
public:
  using View = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = Policy::name;
  Matrix1D_IJK() { }

  Matrix1D_IJK(const RAJA::Index_type rows, const RAJA::Index_type cols, const T& defaultValue = T())
      : rows_(rows), cols_(cols), data_(rows_ * cols_, defaultValue) { }

  RAJA::Index_type getRows() const { return rows_; }
  RAJA::Index_type getCols() const { return cols_; }

  Matrix1D_IJK<T, Policy> operator*(Matrix1D_IJK<T, Policy>& rhs) {
    assert(rhs.cols_ == rows_ && rhs.rows_ == cols_);

    auto result = Matrix1D_IJK(rows_, cols_, T());

    View lhsView(&data_[0], rows_, cols_);
    View rhsView(&rhs.data_[0], rows_, cols_);
    View resultView(&result.data_[0], rows_, cols_);

    RAJA::forall<typename Policy::EXEC>(
      RAJA::RangeSegment(0, rows_),
      [=](RAJA::Index_type row) {
          for (RAJA::Index_type col = 0; col < cols_; ++col) {
              T tmp = 0;
              for (RAJA::Index_type k = 0; k < cols_; ++k) {
                  tmp += lhsView(row, k) * rhsView(k, col);
              }
              resultView(row, col) = tmp;
          }
      }
    );

    return result;
  }


  T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) {
    return data_[row * cols_ + col];
  }

  const T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) const {
    return data_[row * cols_ + col];
  }
};

template <typename T, typename Policy>
class Matrix1D_IKJ {
  RAJA::Index_type rows_;
  RAJA::Index_type cols_;
  std::vector<T> data_;
public:
  using View = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
  static constexpr char* name = Policy::name;
  Matrix1D_IKJ() { }

  Matrix1D_IKJ(const RAJA::Index_type rows, const RAJA::Index_type cols, const T& defaultValue = T())
      : rows_(rows), cols_(cols), data_(rows_ * cols_, defaultValue) { }

  RAJA::Index_type getRows() const { return rows_; }
  RAJA::Index_type getCols() const { return cols_; }

  Matrix1D_IKJ<T, Policy> operator*(Matrix1D_IKJ<T, Policy>& rhs) {
    assert(rhs.cols_ == rows_ && rhs.rows_ == cols_);

    auto result = Matrix1D_IKJ(rows_, cols_, T());

    View lhsView(&data_[0], rows_, cols_);
    View rhsView(&rhs.data_[0], rows_, cols_);
    View resultView(&result.data_[0], rows_, cols_);

    RAJA::forall<typename Policy::EXEC>(
      RAJA::RangeSegment(0, rows_),
      [=](RAJA::Index_type row) {
          for (RAJA::Index_type k = 0; k < cols_; ++k) {
              for (RAJA::Index_type col = 0; col < cols_; ++col) {
                  resultView(row, col) += lhsView(row, k) * rhsView(k, col);
              }
          }
      }
    );

    return result;
  }

  T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) {
    return data_[row * cols_ + col];
  }

  const T& operator()(const RAJA::Index_type row, const RAJA::Index_type col) const {
    return data_[row * cols_ + col];
  }
};

struct SerialPolicy1D {
  using EXEC = RAJA::simd_exec;
  static constexpr char* name = "Serial";
};

struct OmpPolicy1D {
  using EXEC = RAJA::omp_parallel_for_exec;
  static constexpr char* name = "OMP";
};

struct AgencyPolicy1D {
  using EXEC = RAJA::agency_parallel_exec;
  static constexpr char* name = "Agency";
};

struct AgencyOMPPolicy1D {
  using EXEC = RAJA::agency_omp_parallel_exec;
  static constexpr char* name = "AgencyOMP";
};

template <typename T>
std::vector<T> mult(const std::vector<T>& lhs, const std::vector<T>& rhs, 
                    const std::size_t rows, const std::size_t cols) {
  std::vector<T> resultV(rows * cols, 0);

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      T tmp = 0;
      for (std::size_t k = 0; k < rows; ++k) {
        tmp += lhs[row * cols + k] * rhs[k * cols + col];
      }
      resultV[row * cols + col] = tmp;
    }
  }

  return resultV;
}

template <typename T>
std::vector<T> multPERM(const std::vector<T>& lhs, const std::vector<T>& rhs, 
                    const std::size_t rows, const std::size_t cols) {
  std::vector<T> resultV(rows * cols, 0);

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t k = 0; k < rows; ++k) {
      for (std::size_t col = 0; col < cols; ++col) {
        resultV[row * cols + col] += lhs[row * cols + k] * rhs[k * cols + col];
      }
    }
  }

  return resultV;
}

template <typename T>
std::vector<T> multOMP(const std::vector<T>& lhs, const std::vector<T>& rhs, 
                    const std::size_t rows, const std::size_t cols) {
  std::vector<T> resultV(rows * cols, 0);

  #pragma omp parallel for
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      T tmp = 0;
      for (std::size_t k = 0; k < cols; ++k) {
        tmp += lhs[row * cols + k] * rhs[k * cols + col];
      }
      resultV[row * cols + col] = tmp;
    }
  }

  return resultV;
}

template <typename T>
std::vector<T> multOMPPERM(const std::vector<T>& lhs,
                       const std::vector<T>& rhs,
                       const std::size_t rows,
                       const std::size_t cols) {
  std::vector<T> resultV(rows * cols, 0);

  #pragma omp parallel for
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t k = 0; k < rows; ++k) {
      for (std::size_t col = 0; col < cols; ++col) {
        resultV[row * cols + col] += lhs[row * cols + k] * rhs[k * cols + col];
      }
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
void runTimingTest(MATRIX& left, MATRIX& right, 
                   const std::vector<double>& result, 
                   const unsigned numTrials) {
    cali::Annotation timing_test("mode");
    cali::Annotation::Guard guard(timing_test);
    timing_test.set(MATRIX::name);
    cali::Annotation iteration("iteration");
    cali::Annotation::Guard itGuard(iteration);
    for (RAJA::Index_type i = 0; i < numTrials; ++i) {
      iteration.set(i);
      auto actualResult = left * right;
      iteration.set("test");
      checkResult(actualResult, result);
    }
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
void checkResult(const std::vector<double>& actual, const std::vector<double>& expected) {
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::ostringstream os;
            os << "Invalid value at [" << i << "]. Was " << actual[i] << " but expected " << expected[i] << ".\n";
            throw std::runtime_error(os.str());
        }
    }
}

template <typename Serial, typename OMP, typename Agency, typename AgencyOMP>
void doTest(const std::string& perm, const size_t size, const size_t depth)
{
    constexpr static const std::size_t NUM_TRIALS = 10;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0, 5);

    const size_t rows = size;
    const size_t cols = size;

    Serial serial_left(rows, cols);
    Serial serial_right(rows, cols);
    
    OMP omp_left(rows, cols);
    OMP omp_right(rows, cols);

    Agency agency_left(rows, cols);
    Agency agency_right(rows, cols);

    AgencyOMP agencyomp_left(rows, cols);
    AgencyOMP agencyomp_right(rows, cols);

    std::vector<double> control_left(rows * cols);
    std::vector<double> control_right(rows * cols);

    std::vector<double> controlomp_left(rows * cols);
    std::vector<double> controlomp_right(rows * cols);

    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        double first = dist(gen), second = dist(gen);

        serial_left(i, j) = first;
        serial_right(i, j) = second;

        omp_left(i, j) = first;
        omp_right(i, j) = second;

        agency_left(i, j) = first;
        agency_right(i, j) = second;

        agencyomp_left(i, j) = first;
        agencyomp_right(i, j) = second;

        control_left[i * cols + j] = first;
        control_right[i * cols + j] = second;

        controlomp_left[i * cols + j] = first;
        controlomp_right[i * cols + j] = second;
      }
    }

    auto control = cali::Annotation("mode");
    control.set("control");
    std::vector<double> resultV;
    auto iteration = cali::Annotation("iteration");
    for (RAJA::Index_type i = 0; i < NUM_TRIALS; ++i) {
      iteration.set(i);
      if (perm == "IJK") {
        resultV = mult(control_left, control_right, rows, cols);
      } else {
        resultV = multPERM(control_left, control_right, rows, cols);
      }
    }
    iteration.end();
    control.end();
    std::cout << "\t\t\tCompleted control serial without error.\n";

    auto omp = cali::Annotation("mode");
    omp.set("rawOMP");
    std::vector<double> ompResult;
    auto ompiteration = cali::Annotation("iteration");
    for (RAJA::Index_type i = 0; i < NUM_TRIALS; ++i) {
      iteration.set(i);
      if (perm == "IJK") {
        ompResult = multOMP(control_left, control_right, rows, cols);
      } else {
        ompResult = multOMPPERM(control_left, control_right, rows, cols);
      }
    }
    ompiteration.end();
    omp.end();
    checkResult(ompResult, resultV);
    std::cout << "\t\t\tCompleted control omp without error.\n";

    runTimingTest(serial_left, serial_right, resultV, NUM_TRIALS);
    std::cout << "\t\t\tCompleted matrix multiplication serial style without error.\n";

    runTimingTest(omp_left, omp_right, ompResult, NUM_TRIALS);
    std::cout << "\t\t\tCompleted matrix multiplication omp style without error." << std::endl;

    runTimingTest(agency_left, agency_right, ompResult, NUM_TRIALS);
    std::cout << "\t\t\tCompleted matrix multiplication agency style without error." << std::endl;

    runTimingTest(agencyomp_left, agencyomp_right, ompResult, NUM_TRIALS);
    std::cout << "\t\t\tCompleted matrix multiplication agency_omp style without error." << std::endl;
}

int main(int argc, char** argv) {
  constexpr static std::size_t numSizes = 10;

  cali::Annotation size("size");
  cali::Annotation::Guard sizeGuard(size);

  for (int sizeIndex = 0; sizeIndex < numSizes; ++sizeIndex) {
    auto rows = interpolateNumberLinearlyOnLogScale(100, 1000, numSizes, sizeIndex);
    size.set(rows);
    std::cout << "Starting size " << rows << "\n";

    cali::Annotation depth("depth");
    cali::Annotation::Guard depthGuard(depth);
    depth.set(1);
    cali::Annotation perm("perm");
    cali::Annotation::Guard  permGuard(perm);
    perm.set("IJK");
    std::cout << "\tStarting depth 1\n"
              << "\t\tStarting Permutation IJK\n";
    doTest<Matrix1D_IJK<double, SerialPolicy1D>, 
           Matrix1D_IJK<double, OmpPolicy1D>, 
           Matrix1D_IJK<double, AgencyPolicy1D>,
           Matrix1D_IJK<double, AgencyOMPPolicy1D>>("IJK", rows, 1);
    perm.set("IKJ");
    std::cout << "\t\tStarting Permutation IKJ\n";
    doTest<Matrix1D_IKJ<double, SerialPolicy1D>, 
           Matrix1D_IKJ<double, OmpPolicy1D>, 
           Matrix1D_IKJ<double, AgencyPolicy1D>,
           Matrix1D_IKJ<double, AgencyOMPPolicy1D>>("IKJ", rows, 1);

    depth.set(2);
    perm.set("IJK");
    std::cout << "\tStarting depth 2\n"
              << "\t\tStarting Permutation IJK\n";
    doTest<Matrix2D_IJK<double, SerialPolicy2D>, 
           Matrix2D_IJK<double, OmpPolicy2D>, 
           Matrix2D_IJK<double, AgencyPolicy2D>,
           Matrix2D_IJK<double, AgencyOMPPolicy2D>>("IJK", rows, 2);
    perm.set("IKJ");
    std::cout << "\t\tStarting Permutation IKJ\n";
    doTest<Matrix2D_IKJ<double, SerialPolicy2D>, 
           Matrix2D_IKJ<double, OmpPolicy2D>, 
           Matrix2D_IKJ<double, AgencyPolicy2D>,
           Matrix2D_IKJ<double, AgencyOMPPolicy2D>>("IKJ", rows, 2);

    depth.set(3);
    perm.set("IJK");
    std::cout << "\tStarting depth 3\n"
              << "\t\tStarting Permutation IJK\n";
    doTest<Matrix3D<double, SerialPolicy3D>, 
           Matrix3D<double, OmpPolicy3D>, 
           Matrix3D<double, AgencyPolicy3D>,
           Matrix3D<double, AgencyOMPPolicy3D>>("IJK", rows, 3);
    perm.set("IKJ");
    std::cout << "\t\tStarting Permutation IKJ\n";
    doTest<Matrix3D<double, SerialPolicy3D_IKJ>, 
           Matrix3D<double, OmpPolicy3D_IKJ>, 
           Matrix3D<double, AgencyPolicy3D_IKJ>,
           Matrix3D<double, AgencyOMPPolicy3D_IKJ>>("IKJ", rows, 3);
  }
}
