// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <immintrin.h>

#include <iostream>

#include "defines.hpp"
#include "utils/memory.hpp"

FORCE_INLINE void accumulate_one_block(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    uint16_t* __restrict__ result,
    size_t D
) {
    size_t TOTAL = D << 2;  // FAST_SIZE(32) * D / 8
#if defined(__AVX512F__)
    __m512i c, lo, hi, lut, res_lo, res_hi;

    const __m512i lo_mask = _mm512_set1_epi8(0x0f);
    __m512i accu0 = _mm512_setzero_si512();
    __m512i accu1 = _mm512_setzero_si512();
    __m512i accu2 = _mm512_setzero_si512();
    __m512i accu3 = _mm512_setzero_si512();

    for (size_t i = 0; i < TOTAL; i += 128) {
        c = _mm512_load_si512(&codes[i]);
        lut = _mm512_load_si512(&LUT[i]);
        lo = _mm512_and_si512(c, lo_mask);
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));

        c = _mm512_load_si512(&codes[i + 64]);
        lut = _mm512_load_si512(&LUT[i + 64]);
        lo = _mm512_and_si512(c, lo_mask);
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
    }

    __m256i res0 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu0), _mm512_extracti64x4_epi64(accu0, 1)
    );
    __m256i res1 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu1), _mm512_extracti64x4_epi64(accu1, 1)
    );

    res0 = _mm256_sub_epi16(res0, _mm256_slli_epi16(res1, 8));
    __m256i dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(res0, res1, 0x21), _mm256_blend_epi32(res0, res1, 0xF0)
    );
    _mm256_store_si256((__m256i*)result, dis0);

    __m256i res2 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu2), _mm512_extracti64x4_epi64(accu2, 1)
    );
    __m256i res3 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu3), _mm512_extracti64x4_epi64(accu3, 1)
    );

    res2 = _mm256_sub_epi16(res2, _mm256_slli_epi16(res3, 8));
    __m256i dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(res2, res3, 0x21), _mm256_blend_epi32(res2, res3, 0xF0)
    );
    _mm256_store_si256((__m256i*)&result[16], dis1);

#elif defined(__AVX2__)
    __m256i c, lo, hi, lut, res_lo, res_hi;

    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    for (size_t i = 0; i < TOTAL; i += 64) {
        c = _mm256_load_si256(&codes[i]);
        lut = _mm256_load_si256(&LUT[i]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        c = _mm256_load_si256(&codes[i + 32]);
        lut = _mm256_load_si256(&LUT[i + 32]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }

    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    __m256i dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xF0)
    );
    _mm256_store_si256((__m256i*)result, dis0);

    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    __m256i dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0)
    );
    _mm256_store_si256((__m256i*)&result[16], dis1);
#else
    std::cerr << "NO AVX SIMD SUPPORTED!\n";
    abort();
#endif
}

FORCE_INLINE void accumulate_robust(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    uint32_t* __restrict__ result,
    size_t D
) {
    size_t TOTAL = D << 2;  // FAST_SIZE(32) * D / 8
#if defined(__AVX512F__)
    __m512i c, lo, hi, lut, res_lo, res_hi;

    const __m512i lo_mask = _mm512_set1_epi8(0x0f);
    __m512i accu0 = _mm512_setzero_si512();
    __m512i accu1 = _mm512_setzero_si512();
    __m512i accu2 = _mm512_setzero_si512();
    __m512i accu3 = _mm512_setzero_si512();

    for (size_t i = 0; i < TOTAL; i += 128) {
        c = _mm512_load_si512(&codes[i]);
        lut = _mm512_load_si512(&LUT[i]);
        lo = _mm512_and_si512(c, lo_mask);
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));

        c = _mm512_load_si512(&codes[i + 64]);
        lut = _mm512_load_si512(&LUT[i + 64]);
        lo = _mm512_and_si512(c, lo_mask);
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

        res_lo = _mm512_shuffle_epi8(lut, lo);
        res_hi = _mm512_shuffle_epi8(lut, hi);

        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
    }

    __m256i res0 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu0), _mm512_extracti64x4_epi64(accu0, 1)
    );
    __m256i res1 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu1), _mm512_extracti64x4_epi64(accu1, 1)
    );

    res0 = _mm256_sub_epi16(res0, _mm256_slli_epi16(res1, 8));
    __m512i dis0 = _mm512_add_epi32(
        _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(res0, res1, 0x21)),
        _mm512_cvtepu16_epi32(_mm256_blend_epi32(res0, res1, 0xF0))
    );
    _mm512_store_si512(result, dis0);

    __m256i res2 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu2), _mm512_extracti64x4_epi64(accu2, 1)
    );
    __m256i res3 = _mm256_add_epi16(
        _mm512_castsi512_si256(accu3), _mm512_extracti64x4_epi64(accu3, 1)
    );

    res2 = _mm256_sub_epi16(res2, _mm256_slli_epi16(res3, 8));

    __m512i dis1 = _mm512_add_epi32(
        _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(res2, res3, 0x21)),
        _mm512_cvtepu16_epi32(_mm256_blend_epi32(res2, res3, 0xF0))
    );
    _mm512_store_si512(&result[16], dis1);
#endif
}

inline uint32_t accumulate_one_block_high_acc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ LUT,
    const float* __restrict__ onorm,
    float sumq,
    float qnorm,
    float delta,
    int shift,
    float* __restrict__ low_dist,
    float* __restrict__ ip_xb_qprime,
    float one_over_sqrtD,
    float distk,
    size_t D
) {
    __m512i low_mask = _mm512_set1_epi8(0xf);
    __m512i accu[2][4];
    for (size_t _ = 0; _ < 2; _++)
        for (size_t i = 0; i < 4; i++)
            accu[_][i] = _mm512_setzero_si512();

    size_t M = D >> 2;

    // std::cerr << "FastScan YES!" << std::endl;
    for (size_t m = 0; m < M; m += 4) {
        __m512i c = _mm512_load_si512(codes);
        __m512i lo = _mm512_and_si512(c, low_mask);
        __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);

        for (size_t _ = 0; _ < 2; _++) {
            __m512i lut = _mm512_load_si512(LUT);

            __m512i res_lo = _mm512_shuffle_epi8(lut, lo);
            __m512i res_hi = _mm512_shuffle_epi8(lut, hi);

            accu[_][0] = _mm512_add_epi16(accu[_][0], res_lo);
            accu[_][1] = _mm512_add_epi16(accu[_][1], _mm512_srli_epi16(res_lo, 8));

            accu[_][2] = _mm512_add_epi16(accu[_][2], res_hi);
            accu[_][3] = _mm512_add_epi16(accu[_][3], _mm512_srli_epi16(res_hi, 8));

            LUT += 64;
        }
        codes += 64;
    }

    // std::cerr << "FastScan YES!" << std::endl;

    __m512i res[2];
    __m512i dis0[2], dis1[2];

    for (size_t _ = 0; _ < 2; _++) {
        __m256i tmp0 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[_][0]), _mm512_extracti64x4_epi64(accu[_][0], 1)
        );
        __m256i tmp1 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[_][1]), _mm512_extracti64x4_epi64(accu[_][1], 1)
        );
        tmp0 = _mm256_sub_epi16(tmp0, _mm256_slli_epi16(tmp1, 8));

        dis0[_] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp0, tmp1, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp0, tmp1, 0xF0))
        );

        __m256i tmp2 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[_][2]), _mm512_extracti64x4_epi64(accu[_][2], 1)
        );
        __m256i tmp3 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[_][3]), _mm512_extracti64x4_epi64(accu[_][3], 1)
        );
        tmp2 = _mm256_sub_epi16(tmp2, _mm256_slli_epi16(tmp3, 8));

        dis1[_] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp2, tmp3, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp2, tmp3, 0xF0))
        );
    }
    // shift res of high, add res of low
    res[0] = _mm512_add_epi32(dis0[0], _mm512_slli_epi32(dis0[1], 8));
    res[1] = _mm512_add_epi32(dis1[0], _mm512_slli_epi32(dis1[1], 8));

    constexpr float const_bound = 0.58;

    __m512i simd_shift = _mm512_set1_epi32(shift);
    __m512 simd_delta = _mm512_set1_ps(delta);
    // __m512 simd_shift_x_delta = _mm512_set1_ps(delta * shift);
    __m512 simd_sumq_const_bound = _mm512_set1_ps(0.5 * sumq - const_bound);
    __m512 simd_qnorm_over_sqrtD = _mm512_set1_ps(-5 * qnorm * one_over_sqrtD);
    __m512 simd_qnorm_sqr = _mm512_set1_ps(qnorm * qnorm);
    __m512 simd_distk = _mm512_set1_ps(distk);

    uint32_t ret_mask = 0;

    for (size_t i = 0; i < 2; i++) {
        res[i] = _mm512_add_epi32(res[i], simd_shift);
        __m512 tmp = _mm512_cvtepi32_ps(res[i]);
        tmp = _mm512_mul_ps(tmp, simd_delta);
        // tmp = _mm512_add_ps(tmp, simd_shift_x_delta);
        _mm512_store_ps(ip_xb_qprime, tmp);
        // result 1

        tmp = _mm512_sub_ps(tmp, simd_sumq_const_bound);
        // unaligned
        __m512 simd_onorm = _mm512_loadu_ps((onorm + i * 16));

        tmp = _mm512_mul_ps(tmp, simd_qnorm_over_sqrtD);
        tmp = _mm512_mul_ps(tmp, simd_onorm);
        tmp = _mm512_add_ps(tmp, simd_qnorm_sqr);
        tmp = _mm512_add_ps(tmp, _mm512_mul_ps(simd_onorm, simd_onorm));
        ret_mask |= ((uint32_t)_mm512_cmp_ps_mask(tmp, simd_distk, 1) << (i * 16));
        // _mm512_store_ps(low_dist, tmp);

        ip_xb_qprime += 16;
        // low_dist += 16;
    }
    return ret_mask;

    // save (1) <\bar x_b, q'> for further computation; (2) low_dist for checking

    /*
    ||o_r - c||^2 + ||q_r - c||^2 -
        2 * ||o_r - c|| * ||q_r - c|| *  ( <\bar o, q> / 0.8 + bound)
    = ||o_r - c||^2 + ||q_r - c||^2 -
        2.5 * ||o_r - c|| * ||q_r - c|| * ( <\bar o, q> + bound * 0.8)
    = ||o_r - c||^2 + ||q_r - c||^2 -
        2.5 * ||o_r - c|| * ||q_r - c|| * ( <\bar x, q'> + bound * 0.8)
    = ||o_r - c||^2 + ||q_r - c||^2 -
        2.5 * ||o_r - c|| * ||q_r - c|| * ( <2/\sqrt{D} * (x_b - 0.5), q'> + bound * 0.8)
    = ||o_r - c||^2 + ||q_r - c||^2 -
        5/\sqrt{D} * ||o_r - c|| * ||q_r - c|| *
        ( <(x_b - 0.5), q'> + bound * 0.8 * \sqrt{D}/2)
    = ||o_r - c||^2 + ||q_r - c||^2 -
        5/\sqrt{D} * ||o_r - c|| * ||q_r - c|| *
        ( <(x_b - 0.5), q'> + const_bound)
    = ||o_r - c||^2 + ||q_r - c||^2 -
        5/\sqrt{D} * ||o_r - c|| * ||q_r - c|| *
        ( <x_b, q'> - 0.5 * sum_q' + const_bound)
    = ||o_r - c||^2 + ||q_r - c||^2 -
        5/\sqrt{D} * ||o_r - c|| * ||q_r - c|| *
        ( <x_b, \delta * q_s> - 0.5 * sum_q' + const_bound)
    = ||o_r - c||^2 + ||q_r - c||^2 -
        5/\sqrt{D} * ||o_r - c|| * ||q_r - c|| *
        ( \delta * <x_b, q_s> - 0.5 * sum_q' + const_bound)
    */
}