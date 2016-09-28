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

enum Policy {sequential, parallel};

template <typename T>
class Matrix {
    int rows_;
    int cols_;
    std::vector<T> data_;
    Policy policy_;
public:
    Matrix() { }

    Matrix(const int rows, const int cols, const Policy policy, const T& defaultValue = T())
            : rows_(rows), cols_(cols), data_(rows_ * cols_, defaultValue), policy_(policy) { }

    Matrix<T> operator*(Matrix<T>& rhs) {
        assert(cols_ == rhs.rows_);

        auto result = Matrix<T>(rows_, rhs.cols_, policy_, T());

        T* lhs_ptr = data_.data();
        T* rhs_ptr = rhs.data_.data();
        T* result_ptr = result.data_.data();

        switch (policy_) {
            case sequential:
                agency::bulk_invoke(agency::seq(rows_*rhs.cols_),
                                    [=](agency::sequenced_agent& self) {
                                        int row = self.index() / rhs.cols_;
                                        int col = self.index() % rhs.cols_;

                                        for (int k = 0; k < cols_; ++k) {
                                            result_ptr[rhs.cols_ * row + col] += lhs_ptr[cols_ * row + k] *
                                                                                 rhs_ptr[rhs.cols_ * k + col];
                                        }
                                    });
                break;
            case parallel:
                agency::bulk_invoke(agency::par(rows_*rhs.cols_),
                                    [=](agency::parallel_agent& self) {
                                        int row = self.index() / rhs.cols_;
                                        int col = self.index() % rhs.cols_;

                                        for (int k = 0; k < cols_; ++k) {
                                            result_ptr[rhs.cols_ * row + col] += lhs_ptr[cols_ * row + k] *
                                                                                 rhs_ptr[rhs.cols_ * k + col];
                                        }
                                    });
        }
        return result;
    }

    int getRows() const { return rows_; }
    int getCols() const { return cols_; }
    char* getPolicyName() {
        switch (policy_) {
            case sequential:
                return "sequential";
            case parallel:
                return "parallel";
        }
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
        auto rows = interpolateNumberLinearlyOnLogScale(10, 100, numSizes, sizeIndex);
        auto cols = rows;
        size.set(rows);
        std::cout << "Starting size " << rows << "\n";

        auto init = cali::Annotation("initialization").begin();
        auto matrix1 = Matrix<double>(rows, cols, sequential);
        auto matrix2 = Matrix<double>(rows, cols, sequential);
        auto matrix3 = Matrix<double>(rows, cols, parallel);
        auto matrix4 = Matrix<double>(rows, cols, parallel);

        auto control1 = std::vector<double>(rows * cols);
        auto control2 = std::vector<double>(rows * cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
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
        auto resultV = mult(control1, rows, cols, control2, rows, cols);
        control.end();
        try {
            runTimingTest(matrix1, matrix2, resultV, NUM_TRIALS);
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
            return 1;
        }

        std::cout << "Completed sequential matrix multiplication without error.\n";

        try {
            runTimingTest(matrix3, matrix4, resultV, NUM_TRIALS);
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
            return 1;
        }

        std::cout << "Completed parallel matrix multiplication without error." << std::endl;
    }
    size.end();
}
