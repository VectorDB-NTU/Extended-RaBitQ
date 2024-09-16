#pragma once

#include <stdint.h>

#include "third/Eigen/Dense"

#define FORCE_INLINE inline __attribute__((always_inline))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define lowbit(x) (x & (-x))
#define bit_id(x) (__builtin_popcount(x - 1))

constexpr size_t FAST_SIZE = 32;

using PID = uint32_t;
using pair_di = std::pair<double, int>;
using FloatRowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using IntRowMat = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using UintRowMat = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DoubleRowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Candidate {
    PID id;
    float distance;

    Candidate() = default;
    Candidate(PID id, float distance) : id(id), distance(distance) {}

    bool operator<(const Candidate& other) const { return distance < other.distance; }

    bool operator>(const Candidate& other) const { return !(*this < other); }
};

struct ExFactor {
    float xipnorm;
};
