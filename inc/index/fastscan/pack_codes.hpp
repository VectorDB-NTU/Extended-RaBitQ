
// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

template <typename T, class TA>
static inline void get_matrix_column(
    T* src, size_t m, size_t n, size_t i, size_t j, TA& dest
) {
    for (size_t k = 0; k < dest.size(); k++) {
        if (k + i < m) {
            dest[k] = src[(k + i) * n + j];
        } else {
            dest[k] = 0;
        }
    }
}

// ==============================================================
// pack 32 quantization codes in a batch from the quantization
// codes represented by a sequence of uint8_t variables
// ==============================================================
void pack_codes(size_t B, const uint8_t* codes, size_t ncode, uint8_t* blocks) {
    size_t ncode_pad = (ncode + 31) / 32 * 32;
    size_t M = B / 4;
    const uint8_t bbs = 32;
    memset(blocks, 0, ncode_pad * M / 2);

#if defined(HIGH_ACC_FAST_SCAN)
    // high acc fastscan
    // avx2
    // const uint8_t perm0[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    // avx512
    const uint8_t perm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

#else
    const uint8_t perm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
#endif
    uint8_t* codes2 = blocks;
    for (size_t blk = 0; blk < ncode_pad; blk += bbs) {
        // enumerate m
        for (size_t m = 0; m < M; m += 2) {
            std::array<uint8_t, 32> c, c0, c1;
            get_matrix_column(codes, ncode, M / 2, blk, m / 2, c);
            for (int j = 0; j < 32; j++) {
                c0[j] = c[j] & 15;
                c1[j] = c[j] >> 4;
            }
            for (int j = 0; j < 16; j++) {
                uint8_t d0, d1;
                d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                codes2[j] = d0;
                codes2[j + 16] = d1;
            }
            codes2 += 32;
        }
    }
}

// ==============================================================
// pack 32 quantization codes in a batch from the quantization
// codes represented by a sequence of uint64_t variables
// ==============================================================
void pack_codes(size_t B, const uint64_t* binary_code, size_t ncode, uint8_t* blocks) {
    size_t ncode_pad = (ncode + 31) / 32 * 32;
    memset(blocks, 0, ncode_pad * sizeof(uint8_t));

    uint8_t* binary_code_8bit = new uint8_t[ncode_pad * B / 8];
    memcpy(binary_code_8bit, binary_code, ncode * B / 64 * sizeof(uint64_t));

    for (size_t i = 0; i < ncode; i++)
        for (size_t j = 0; j < B / 64; j++)
            for (size_t k = 0; k < 4; k++)
                swap(
                    binary_code_8bit[i * B / 8 + 8 * j + k],
                    binary_code_8bit[i * B / 8 + 8 * j + 8 - k - 1]
                );

    for (uint32_t i = 0; i < ncode * B / 8; i++) {
        uint8_t v = binary_code_8bit[i];
        uint8_t x = (v >> 4);
        uint8_t y = (v & 15);
        binary_code_8bit[i] = (y << 4 | x);
    }
    pack_codes(B, binary_code_8bit, ncode, blocks);
    delete[] binary_code_8bit;
}