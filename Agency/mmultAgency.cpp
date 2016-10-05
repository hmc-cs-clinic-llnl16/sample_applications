#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cassert>

#include <iostream>
#include <vector>
#include <random>
#include <sstream>

#include "agency/agency.hpp"

#include "caliper/Annotation.h"

enum Policy {ser, par};

template <typename T>
class Matrix {
    int rows_;
    int cols_;
    int workers_;
    std::vector<T> data_;
    Policy policy_;
public:
    Matrix() { }

    Matrix(const int rows, const int cols, const int workers, const Policy policy, const T& defaultValue = T())
            : rows_(rows), cols_(cols), workers_(workers), data_(rows_ * cols_, defaultValue), policy_(policy) { }

    Matrix<T> operator*(Matrix<T>& rhs) {
        assert(cols_ == rhs.rows_);

        auto result = Matrix<T>(rows_, rhs.cols_, workers_, policy_, T());

        T* lhs_ptr = data_.data();
        T* rhs_ptr = rhs.data_.data();
        T* result_ptr = result.data_.data();

        int tile_size = rows_/workers_;

        switch (policy_) {
            case ser:
                agency::bulk_invoke(agency::seq(workers_),
                                    [=](agency::sequenced_agent& self) {
                                        for (int row = self.index()*tile_size; (row < (self.index()+1)*tile_size); ++row) {
                                            for (int col = 0; col < rhs.cols_; ++col) {
                                                for (int k = 0; k < cols_; ++k) {
                                                    result_ptr[rhs.cols_ * row + col] += lhs_ptr[cols_ * row + k] *
                                                                                         rhs_ptr[rhs.cols_ * k + col];
                                                }
                                            }
                                        }
                                    });
                for (int row = workers_*tile_size; (row < rows_); ++row) {
                    for (int col = 0; col < rhs.cols_; ++col) {
                        for (int k = 0; k < cols_; ++k) {
                            result_ptr[rhs.cols_ * row + col] += lhs_ptr[cols_ * row + k] *
                                                                 rhs_ptr[rhs.cols_ * k + col];
                        }
                    }
                }
                break;
            case par:
                agency::bulk_invoke(agency::par(workers_),
                                    [=](agency::parallel_agent& self) {
                                        for (int row = self.index()*tile_size; (row < (self.index()+1)*tile_size); ++row) {
                                            for (int col = 0; col < rhs.cols_; ++col) {
                                                for (int k = 0; k < cols_; ++k) {
                                                    result_ptr[rhs.cols_ * row + col] += lhs_ptr[cols_ * row + k] *
                                                                                         rhs_ptr[rhs.cols_ * k + col];
                                                }
                                            }
                                        }
                                    });
                for (int row = workers_*tile_size; (row < rows_); ++row) {
                    for (int col = 0; col < rhs.cols_; ++col) {
                        for (int k = 0; k < cols_; ++k) {
                            result_ptr[rhs.cols_ * row + col] += lhs_ptr[cols_ * row + k] *
                                                                 rhs_ptr[rhs.cols_ * k + col];
                        }
                    }
                }

        }
        return result;
    }

    int getRows() const { return rows_; }
    int getCols() const { return cols_; }

    const char* getPolicyName() {
        std::stringstream stringStream;
        switch (policy_) {
            case ser:
                stringStream << "Serial";
                break;
            case par:
                stringStream << "Parallel";
        }
        stringStream << workers_ << "Worker";
        return stringStream.str().c_str();
    }


    T& operator()(const int row, const int col) {
        return data_[row * cols_ + col];
    }

    const T& operator()(const int row, const int col) const {
        return data_[row * cols_ + col];
    }
};

template <typename T>
std::vector<T> mult(const std::vector<T>& lhs, const int lhs_rows, const int lhs_cols,
                    const std::vector<T>& rhs, const int rhs_rows, const int rhs_cols) {
    std::vector<T> resultV(rhs_rows * lhs_cols, 0);

    for (int row = 0; row < lhs_rows; ++row) {
        for (int col = 0; col < rhs_cols; ++col) {
            T result = 0;
            for (int k = 0; k < lhs_cols; ++k) {
                result += lhs[row * lhs_cols + k] * rhs[k * rhs_cols + col];
            }
            resultV[row * rhs_cols + col] = result;
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
    cali::Annotation::Guard timing_test(cali::Annotation(left.getPolicyName()).begin());
    auto iteration = cali::Annotation("iteration");
    for (int i = 0; i < numTrials; ++i) {
        std::cout << "Started iteration " << i << " of type " << left.getPolicyName() << "\n";
        iteration.set(i);
        auto actualResult = left * right;
        iteration.set("test");
        checkResult(actualResult, result);
    }
    iteration.end();
}

template <typename MATRIX>
void checkResult(const MATRIX& actual, const std::vector<double>& expected) {
    for (int i = 0; i < actual.getRows(); ++i) {
        for (int j = 0; j < actual.getCols(); ++j) {
            if (actual(i, j) != expected[i * actual.getCols() + j]) {
                std::ostringstream os;
                os << "Invalid value at: (" << i << ", " << j << "). Was " << actual(i, j) << " but expected " << expected[i * actual.getCols()  + j] << "\n";
                throw std::runtime_error(os.str());
            }
        }
    }
}

int main(int argc, char** argv) {
    constexpr static int numSizes = 10;
    constexpr static int NUM_TRIALS = 10;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0, 5);

    auto size = cali::Annotation("size");
    for (int sizeIndex = 0; sizeIndex < numSizes; ++sizeIndex) {
        auto rows = interpolateNumberLinearlyOnLogScale(20, 100, numSizes, sizeIndex);
        auto cols = rows;
        size.set(rows);
        std::cout << "Starting size " << rows << "\n";

        auto init = cali::Annotation("initialization").begin();
        auto matrix1Ser = Matrix<double>(rows, cols, 1, ser);
        auto matrix1Par = Matrix<double>(rows, cols, 1, par);
        auto matrix2Ser = Matrix<double>(rows, cols, 4, ser);
        auto matrix2Par = Matrix<double>(rows, cols, 4, par);
        auto matrix4Ser = Matrix<double>(rows, cols, 16, ser);
        auto matrix4Par = Matrix<double>(rows, cols, 16, par);
        auto matrixRight = Matrix<double>(rows, cols, 1, ser);

        auto control1 = std::vector<double>(rows * cols);
        auto control2 = std::vector<double>(rows * cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double first = dist(gen), second = dist(gen);
                matrix1Ser(i, j) = first;
                matrix1Par(i, j) = first;
                matrix2Ser(i, j) = first;
                matrix2Par(i, j) = first;
                matrix4Ser(i, j) = first;
                matrix4Par(i, j) = first;
                control1[i * cols + j] = first;

                matrixRight(i, j) = second;
                control2[i * cols + j] = second;
            }
        }
        init.end();

        cali::Annotation::Guard control(cali::Annotation("control").begin());
        auto iteration = cali::Annotation("iteration");
        iteration.set(0);
        auto resultV = mult(control1, rows, cols, control2, rows, cols);
        for (int i=1; i<NUM_TRIALS; ++i) {
            iteration.set(i);
            auto resultV = mult(control1, rows, cols, control2, rows, cols);
        }
        iteration.end();

        try {
            runTimingTest(matrix1Ser, matrixRight, resultV, NUM_TRIALS);
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
            return 1;
        }

        std::cout << "Completed 1 worker serial matrix multiplication without error.\n";

        try {
            runTimingTest(matrix1Par, matrixRight, resultV, NUM_TRIALS);
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
            return 1;
        }

        std::cout << "Completed 1 worker parallel matrix multiplication without error.\n";

        try {
            runTimingTest(matrix2Ser, matrixRight, resultV, NUM_TRIALS);
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
            return 1;
        }

        std::cout << "Completed 2 worker serial matrix multiplication without error.\n";

        try {
            runTimingTest(matrix2Par, matrixRight, resultV, NUM_TRIALS);
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
            return 1;
        }

        std::cout << "Completed 2 worker parallel matrix multiplication without error.\n";

        try {
            runTimingTest(matrix4Ser, matrixRight, resultV, NUM_TRIALS);
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
            return 1;
        }

        std::cout << "Completed 4 worker serial matrix multiplication without error.\n";

        try {
            runTimingTest(matrix4Par, matrixRight, resultV, NUM_TRIALS);
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
            return 1;
        }

        std::cout << "Completed 4 worker parallel matrix multiplication without error." << std::endl;
    }
    size.end();
}
