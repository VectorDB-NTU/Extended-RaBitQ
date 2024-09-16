#pragma once

#include <fstream>
#include <iostream>

#include "defines.hpp"
#include "utils/tools.hpp"

class Rotator {
   private:
    size_t D;       // Padded dimension
    FloatRowMat P;  // Rotation Maxtrix
   public:
    explicit Rotator(uint32_t dim) : D(rd_up_to_multiple_of(dim, 64)) {
        FloatRowMat RAND(FloatRowMat::Random(D, D));
        Eigen::HouseholderQR<FloatRowMat> qr(RAND);
        FloatRowMat Q = qr.householderQ();
        this->P = Q.transpose();  // inverse of Q = Q.T
    }

    explicit Rotator() {}

    Rotator& operator=(const Rotator& other) {
        this->D = other.D;
        this->P = other.P;
        return *this;
    }

    size_t size() const { return D; }

    /*
     * Load the rotation matrix from disk
     */
    void load(std::ifstream& input) {
        float element;
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                input.read((char*)&element, sizeof(float));
                P(i, j) = element;
            }
        }
    }

    /*
     * Save the rotation matrix to disk
     */
    void save(std::ofstream& output) const {
        float element;
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                element = P(i, j);
                output.write((char*)&element, sizeof(float));
            }
        }
    }

    /*
     * Rotate Matrix A and store the result in RAND_A
     */
    void rotate(const FloatRowMat& A, FloatRowMat& RAND_A) const {
        // Note that Eigen store Matrix by columns by default
        RAND_A = A * P;
    }
};