#pragma once

#include <immintrin.h>
#include <stdint.h>

#include "defines.hpp"
#include "index/Cluster.hpp"
#include "index/Pool.hpp"
#include "index/Quantizer.hpp"
#include "index/fastscan/FastScan.hpp"
#include "utils/memory.hpp"
#include "utils/space.hpp"

class Searcher {
   private:
    constexpr static size_t BQUERY = 14;  // num bits of quantizing query
    size_t D;
    size_t TABLE_LENGTH;
    const float* query = nullptr;
    float* residual = nullptr;
    uint16_t* byte_query = nullptr;
    uint8_t* LUT_upper = nullptr;
    uint8_t* LUT_lower = nullptr;
    uint16_t* LUT_total = nullptr;
    uint32_t PORTABLE_ALIGN64 result_upper[FAST_SIZE];
    uint32_t PORTABLE_ALIGN64 result_lower[FAST_SIZE];
    float PORTABLE_ALIGN64 lower_distances[FAST_SIZE];
    float PORTABLE_ALIGN64 rabitq_ip[FAST_SIZE];
    const DataQuantizer& DQ;
    float (*IP_FUNC
    )(const float* __restrict__, const uint8_t* __restrict__, size_t
    );  // Function to get ip between query and long code
    float vl = 0;
    float vr = 0;
    float width = 0;
    float half_sumresidual = 0;
    int FAC_RESCALE = 0;

    inline void preparing(const float*);
    inline void pack_LUT();
    FORCE_INLINE void scan_one_block(
        uint8_t*,
        float*,
        PID*,
        float,
        float,
        float&,
        const Cluster&,
        ResultPool&,
        size_t,
        size_t
    );

   public:
    explicit Searcher(const float* q, size_t d, size_t ex_bits, const DataQuantizer& dq)
        : D(d), TABLE_LENGTH(D / 4 * 16), query(q), DQ(dq), FAC_RESCALE(1 << ex_bits) {
        residual = memory::align_mm<64, float>(D * sizeof(float));
        byte_query = memory::align_mm<64, uint16_t>(D * sizeof(uint16_t));
        LUT_upper = memory::align_mm<64, uint8_t>(TABLE_LENGTH * sizeof(uint8_t));
        LUT_lower = memory::align_mm<64, uint8_t>(TABLE_LENGTH * sizeof(uint8_t));
        LUT_total = memory::align_mm<64, uint16_t>(TABLE_LENGTH * sizeof(uint16_t));
        if (ex_bits == 8) {
            IP_FUNC = IP16_fxu8;
        } else if (ex_bits == 4) {
            IP_FUNC = IP32_fxu4;
        } else if (ex_bits == 6) {
            IP_FUNC = IP64_fxu6;
        } else if (ex_bits == 2) {
            IP_FUNC = IP64_fxu2;
        } else if (ex_bits == 3) {
            IP_FUNC = IP64_fxu3;
        } else if (ex_bits == 7) {
            IP_FUNC = IP64_fxu7;
        }
    }

    ~Searcher() {
        std::free(residual);
        std::free(byte_query);
        std::free(LUT_upper);
        std::free(LUT_lower);
        std::free(LUT_total);
    }

    void search_cluster(
        const Cluster& cur_cluster, const float* centroid, float sqr_y, ResultPool& KNNs
    ) {
        preparing(centroid);

        size_t ITER = cur_cluster.iter();
        size_t REMAIN = cur_cluster.remain();

        uint8_t* block = cur_cluster.first_block();
        PID* ids = cur_cluster.ids();
        float distk = KNNs.distk();
        float y = std::sqrt(sqr_y);

        /* Compute distances block by block */
        for (size_t i = 0; i < ITER; ++i) {
            float* block_fac = DQ.block_factor(block);
            scan_one_block(
                block, block_fac, ids, sqr_y, y, distk, cur_cluster, KNNs, i, FAST_SIZE
            );
            block = DQ.next_block(block_fac);
            ids = &ids[FAST_SIZE];
        }

        if (REMAIN > 0) {
            float* block_fac = DQ.block_factor(block);
            scan_one_block(
                block, block_fac, ids, sqr_y, y, distk, cur_cluster, KNNs, ITER, REMAIN
            );
        }
    }
};

/**
 * @brief Preparing data before search, including: 1) quantize query 2) pack LUTs
 *
 * @param centroid Pointer to current scanned centroid vector
 */
inline void Searcher::preparing(const float* centroid) {
    half_sumresidual = 0.5 * minus_sum16(query, centroid, residual, D);
    data_range16(residual, vl, vr, this->D);
    width = (vr - vl) / ((1 << BQUERY) - 1);
    scalar_quantize16<uint16_t, 16>(byte_query, residual, vl, width, D);
    pack_LUT();
}

inline void Searcher::pack_LUT() {
    constexpr int pos[16] = {
        3 /*0000*/,
        3 /*0001*/,
        2 /*0010*/,
        3 /*0011*/,
        1 /*0100*/,
        3 /*0101*/,
        2 /*0110*/,
        3 /*0111*/,
        0 /*1000*/,
        3 /*1001*/,
        2 /*1010*/,
        3 /*1011*/,
        1 /*1100*/,
        3 /*1101*/,
        2 /*1110*/,
        3 /*1111*/,
    };
    uint16_t* lut_t = LUT_total;
    uint16_t* byte_q = byte_query;
    for (size_t i = 0; i < TABLE_LENGTH; i += 32) {
        lut_t[0] = 0;
        for (int j = 1; j < 16; ++j) {
            lut_t[j] = lut_t[j - lowbit(j)] + byte_q[pos[j]];
        }
        lut_t += 16;
        byte_q += 4;
        lut_t[0] = 0;
        for (int j = 1; j < 16; ++j) {
            lut_t[j] = lut_t[j - lowbit(j)] + byte_q[pos[j]];
        }
        lut_t += 16;
        byte_q += 4;

        __m512i total = _mm512_load_epi32(&LUT_total[i]);
        __m512i upper16 = _mm512_srli_epi16(total, 8);
        __m256i upper8 = _mm512_cvtepi16_epi8(upper16);
        _mm256_store_epi32(&LUT_upper[i], upper8);
        __m256i lower8 = _mm512_cvtepi16_epi8(total);
        _mm256_store_epi32(&LUT_lower[i], lower8);
    }
}

FORCE_INLINE void Searcher::scan_one_block(
    uint8_t* block,
    float* block_fac,
    PID* ids,
    float sqr_y,
    float y,
    float& distk,
    const Cluster& cur_cluster,
    ResultPool& KNNs,
    size_t scanned_block,
    size_t num_points
) {
    accumulate_robust(block, LUT_upper, result_upper, D);
    accumulate_robust(block, LUT_lower, result_lower, D);
    const float* factor_x2 = DQ.factor_x2(block_fac);
    const float* factor_ip = DQ.factor_ip(block_fac);
    const float* factor_sumxb = DQ.factor_sumxb(block_fac);
    const float* factor_err = DQ.factor_err(block_fac);
#if defined(__AVX512F__)
    __m512 sqr_y_simd = _mm512_set1_ps(sqr_y);
    __m512 y_simd = _mm512_set1_ps(y);
    __m512 width_simd = _mm512_set1_ps(width);
    __m512 vl_simd = _mm512_set1_ps(vl);
    __m512 half_sumresidual_simd = _mm512_set1_ps(half_sumresidual);
    for (size_t j = 0; j < FAST_SIZE; j += 16) {
        __m512 sum_sqr = _mm512_add_ps(_mm512_load_ps(&factor_x2[j]), sqr_y_simd);
        __m512 xbvl = _mm512_mul_ps(_mm512_load_ps(&factor_sumxb[j]), vl_simd);

        __m512 resf = _mm512_cvtepi32_ps(_mm512_add_epi32(
            _mm512_slli_epi32(_mm512_load_epi32(&result_upper[j]), 8),
            _mm512_load_epi32(&result_lower[j])
        ));
        __m512 ip = _mm512_add_ps(_mm512_mul_ps(resf, width_simd), xbvl);

        ip = _mm512_sub_ps(ip, half_sumresidual_simd);
        _mm512_store_ps(&rabitq_ip[j], ip);
        __m512 fac_ip = _mm512_load_ps(&factor_ip[j]);
        ip = _mm512_mul_ps(ip, fac_ip);

        __m512 err = _mm512_mul_ps(_mm512_load_ps(&factor_err[j]), y_simd);
        __m512 lower = _mm512_sub_ps(sum_sqr, _mm512_add_ps(ip, err));
        _mm512_store_ps(&lower_distances[j], lower);
    }
    for (size_t j = 0; j < num_points; ++j) {
        float lower_dist = lower_distances[j];
        if (lower_dist > distk) {
            continue;
        } else {
            PID id = ids[j];
            float sqr_x = factor_x2[j];
            size_t idx = j + scanned_block * FAST_SIZE;
            uint8_t* long_code = cur_cluster.long_code(idx, DQ);
            ExFactor ex_fac = *cur_cluster.ex_factor(idx);
            float ex_dist = sqr_x + sqr_y -
                            ex_fac.xipnorm * (FAC_RESCALE * rabitq_ip[j] +
                                              IP_FUNC(residual, long_code, D) -
                                              (FAC_RESCALE - 1) * half_sumresidual);

            KNNs.insert(id, ex_dist);
            distk = KNNs.distk();
        }
    }
#endif
}