#pragma once

#include <stdint.h>

#include <limits>
#include <vector>

#include "utils/memory.hpp"

struct ResultPool {
   public:
    ResultPool(size_t capacity)
        : ids_(capacity + 1), distances_(capacity + 1), capacity_(capacity) {}

    void insert(PID u, float dist) {
        if (size_ == capacity_ && dist > distances_[size_ - 1]) {
            return;
        }
        size_t lo = find_bsearch(dist);
        std::memmove(&ids_[lo + 1], &ids_[lo], (size_ - lo) * sizeof(PID));
        ids_[lo] = u;
        std::memmove(&distances_[lo + 1], &distances_[lo], (size_ - lo) * sizeof(float));
        distances_[lo] = dist;
        size_ += (size_ < capacity_);
        return;
    }

    float distk() {
        return size_ == capacity_ ? distances_[size_ - 1]
                                  : std::numeric_limits<float>::max();
    }

    void copy_results(PID* KNN) { std::copy(ids_.begin(), ids_.end() - 1, KNN); }

   private:
    std::vector<PID, memory::align_allocator<PID>> ids_;
    std::vector<float, memory::align_allocator<float>> distances_;
    size_t size_ = 0, capacity_;

    size_t find_bsearch(float dist) const {
        size_t lo = 0, len = size_;
        size_t half;
        while (len > 1) {
            half = len >> 1;
            len -= half;
            lo += (distances_[lo + half - 1] < dist) * half;
        }
        return (lo < size_ && distances_[lo] < dist) ? lo+1 : lo;
    }
};