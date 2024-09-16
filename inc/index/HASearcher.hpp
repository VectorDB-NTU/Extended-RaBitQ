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

class HASearcher {
   private:
    constexpr static size_t BQUERY = 14;  // num bits of quantizing query
    size_t D;
    size_t TABLE_LENGTH;
    const float* query = nullptr;
    float* unit_q = nullptr;
    int16_t* quant_query = nullptr;
    uint8_t* HC_LUT = nullptr;
    float PORTABLE_ALIGN64 lower_distances[FAST_SIZE];
    float PORTABLE_ALIGN64 ip_xb_qprime[FAST_SIZE];
    const DataQuantizer& DQ;
    float (*IP_FUNC
    )(const float* __restrict__, const uint8_t* __restrict__, size_t
    );  // Function to get ip between query and long code
    float one_over_sqrtD = 0;
    float delta = 0;
    float sumq = 0;
    int shift = 0;
    int FAC_RESCALE = 0;

    inline void preparing(const float*, float);
    inline void pack_high_acc_LUT();
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
    explicit HASearcher(const float* q, size_t d, size_t ex_bits, const DataQuantizer& dq)
        : D(d)
        , TABLE_LENGTH(D / 4 * 16)
        , query(q)
        , DQ(dq)
        , one_over_sqrtD(1.0 / std::sqrt((float)D))
        , FAC_RESCALE(1 << ex_bits) {
        unit_q = memory::align_mm<64, float>(D * sizeof(float));
        quant_query = memory::align_mm<64, int16_t>(D * sizeof(int16_t));
        HC_LUT = memory::align_mm<64, uint8_t>(TABLE_LENGTH * 2 * sizeof(uint8_t));
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

    ~HASearcher() {
        std::free(unit_q);
        std::free(quant_query);
        std::free(HC_LUT);
    }

    void search_cluster(
        const Cluster& cur_cluster, const float* centroid, float sqr_y, ResultPool& KNNs
    ) {
        float y = std::sqrt(sqr_y);
        preparing(centroid, y);

        size_t ITER = cur_cluster.iter();
        size_t REMAIN = cur_cluster.remain();

        uint8_t* block = cur_cluster.first_block();
        PID* ids = cur_cluster.ids();
        float distk = KNNs.distk();

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
            // scan the last block
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
 * @param y distance from query to centroid
 */
inline void HASearcher::preparing(const float* centroid, float y) {
    this->shift = 0;
    this->sumq = normalize_query16(unit_q, query, centroid, y, D);
    high_acc_quantize16(quant_query, unit_q, delta, D);
    pack_high_acc_LUT();
}

inline void HASearcher::pack_high_acc_LUT() {
    size_t M = D >> 2;
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

    int16_t* quan_query = quant_query;
    for (size_t i = 0; i < M; i++) {
        int PORTABLE_ALIGN64 LUT[16];
        int v_min = 0;

        LUT[0] = 0;
        for (int j = 1; j < 16; j++) {
            LUT[j] = LUT[j - lowbit(j)] + quan_query[pos[j]];
            v_min = (LUT[j] < v_min) ? LUT[j] : v_min;
        }

        // avx2 - 256, avx512 - 512
        constexpr size_t B_regi = 512;
        constexpr size_t B_lane = 128;
        constexpr size_t B_byte = 8;

        constexpr size_t n_lut_per_iter = B_regi / B_lane;
        constexpr size_t n_code_per_iter = 2 * B_regi / B_byte;
        constexpr size_t n_code_per_lane = B_lane / B_byte;

        uint8_t* fill_lo = HC_LUT + i / n_lut_per_iter * n_code_per_iter +
                           (i % n_lut_per_iter) * n_code_per_lane;
        uint8_t* fill_hi = fill_lo + B_regi / B_byte;

        /* shift all the elements in LUT such that they become unsigned integer */
        __m512i lut = _mm512_load_epi32(LUT);
        __m512i tmp = _mm512_sub_epi32(lut, _mm512_set1_epi32(v_min));
        __m128i lo = _mm512_cvtepi32_epi8(tmp);
        __m128i hi = _mm512_cvtepi32_epi8(_mm512_srli_epi32(tmp, 8));
        _mm_store_si128((__m128i*)fill_lo, lo);
        _mm_store_si128((__m128i*)fill_hi, hi);

        // the shifted valued will be finally added back
        shift += v_min;
        quan_query += 4;
    }
}

FORCE_INLINE void HASearcher::scan_one_block(
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
    const float* factor_x = DQ.factor_x2(block_fac);
    uint32_t mask = accumulate_one_block_high_acc(
        block,
        HC_LUT,
        factor_x,
        sumq,
        y,
        delta,
        shift,
        lower_distances,
        ip_xb_qprime,
        one_over_sqrtD,
        distk,
        D
    );

    // The following line is important: the number of num_points is not necessarily 32.
    mask = (mask & ((1 << num_points) - 1));

    // incremental distance computation - V2
    while (mask) {
        uint32_t lb = lowbit(mask);
        uint32_t j = bit_id(lb);
        mask -= lb;
        PID id = ids[j];
        // std::cerr << j << " ";
        float sqr_x = factor_x[j] * factor_x[j];
        size_t idx = j + scanned_block * FAST_SIZE;
        uint8_t* long_code = cur_cluster.long_code(idx, DQ);
        ExFactor ex_fac = *cur_cluster.ex_factor(idx);
        float ex_dist = sqr_x + sqr_y -
                        ex_fac.xipnorm * y *
                            (FAC_RESCALE * ip_xb_qprime[j] + IP_FUNC(unit_q, long_code, D) -
                             (FAC_RESCALE - 0.5) * sumq);

        KNNs.insert(id, ex_dist);
        distk = KNNs.distk();
    }
}