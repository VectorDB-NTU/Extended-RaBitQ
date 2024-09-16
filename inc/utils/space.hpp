#pragma once

#include <bits/stdc++.h>
#include <immintrin.h>

#include <cstdint>

#include "defines.hpp"
#include "utils/memory.hpp"

FORCE_INLINE float reduce_add_f32x8(__m256 x) {
    auto sumh = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
    auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
    return _mm_cvtss_f32(tmp2);
}

FORCE_INLINE float L2Sqr16(
    const float* __restrict__ x, const float* __restrict__ y, size_t L
) {
    float result = 0;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < L; i += 16) {
        __m512 xx = _mm512_loadu_ps(&x[i]);
        __m512 yy = _mm512_loadu_ps(&y[i]);
        __m512 t = _mm512_sub_ps(xx, yy);
        sum = _mm512_fmadd_ps(t, t, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    return result;
#else
    for (size_t i = 0; i < L; ++i) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return result;
#endif
}

FORCE_INLINE float L2Sqr(
    const float* __restrict__ x, const float* __restrict__ y, size_t L
) {
    size_t num16 = L - (L & 0b1111);
    float result = L2Sqr16(x, y, num16);
    for (size_t i = num16; i < L; ++i) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return result;
}

/* Compute L2sqr to origin */
FORCE_INLINE float L2Sqr(const float* __restrict__ x, size_t L) {
    float result = 0;
#if defined(__AVX512F__)
    size_t num16 = L - (L & 0b1111);
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num16; i += 16) {
        __m512 xx = _mm512_loadu_ps(&x[i]);
        sum = _mm512_fmadd_ps(xx, xx, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (size_t i = num16; i < L; ++i) {
        float tmp = x[i];
        result += tmp * tmp;
    }
    return result;
#else
    for (size_t i = 0; i < L; ++i) {
        float tmp = x[i];
        result += tmp * tmp;
    }
    return result;
#endif
}

FORCE_INLINE float IP(const float* x, const float* y, size_t L) {
    float result = 0;
#if defined(__AVX512F__)
    size_t num16 = L - (L & 0b1111);
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num16; i += 16) {
        __m512 xx = _mm512_loadu_ps(&x[i]);
        __m512 yy = _mm512_loadu_ps(&y[i]);
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (size_t i = num16; i < L; ++i) {
        result += x[i] * y[i];
    }
    return result;
#else
    for (size_t i = 0; i < L; ++i) {
        result += x[i] * y[i];
    }
    return result;
#endif
}

/**
 * @brief Inner produce between float vec and uint8 vec, length must be a mutiple of 16
 *
 * @param x Float vec
 * @param y Uint8 vec
 * @param L Length of vecs
 * @return FORCE_INLINE
 */
FORCE_INLINE float IP16_fxu8(
    const float* __restrict__ x, const uint8_t* __restrict__ y, size_t L
) {
    float result = 0;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < L; i += 16) {
        __m512 xx = _mm512_load_ps(&x[i]);
        __m512 yy =
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_load_si128((__m128i*)&y[i])));
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    result = _mm512_reduce_add_ps(sum);
#else
    for (size_t i = 0; i < L; i += 4) {
        result += x[i] * static_cast<float>(y[i]);
        result += x[i + 1] * static_cast<float>(y[i + 1]);
        result += x[i + 2] * static_cast<float>(y[i + 2]);
        result += x[i + 3] * static_cast<float>(y[i + 3]);
    }
#endif
    return result;
}

FORCE_INLINE float IP_fxu8(
    const float* __restrict__ x, const uint8_t* __restrict__ y, size_t L
) {
    size_t num16 = L - (L & 0b1111);
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num16; i += 16) {
        __m512 xx = _mm512_loadu_ps(&x[i]);
        __m512 yy =
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)&y[i])));
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    float result = _mm512_reduce_add_ps(sum);

    for (size_t i = num16; i < L; ++i) {
        result += x[i] * static_cast<float>(y[i]);
    }
    return result;
}

FORCE_INLINE float IP32_fxu4(
    const float* __restrict__ x, const uint8_t* __restrict__ y, size_t D
) {
    __m128i mask = _mm_set1_epi8(0b1111);
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < D; i += 32) {
        __m128i a8 = _mm_load_epi32(&y[i / 2]);
        __m128i b8 = a8;
        __m512 x1 = _mm512_load_ps(&x[i]);
        __m512 x2 = _mm512_load_ps(&x[i + 16]);

        // get lower(0 to 15) and upper(16 to 31) 4 bits
        a8 = _mm_and_si128(a8, mask);
        b8 = _mm_and_si128(_mm_srli_epi16(b8, 4), mask);

        __m512 af = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a8));
        sum = _mm512_fmadd_ps(af, x1, sum);
        __m512 bf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b8));
        sum = _mm512_fmadd_ps(bf, x2, sum);
    }
    return _mm512_reduce_add_ps(sum);
}

FORCE_INLINE float IP_fxu4(
    const float* __restrict__ x, const uint8_t* __restrict__ y, size_t D
) {
    __m128i mask = _mm_set1_epi8(0b1111);
    __m512 sum = _mm512_setzero_ps();
    size_t num32 = D - (D & 0x1F);
    for (size_t i = 0; i < num32; i += 32) {
        __m128i a8 = _mm_loadu_epi32(&y[i / 2]);
        __m128i b8 = a8;
        __m512 x1 = _mm512_loadu_ps(&x[i]);
        __m512 x2 = _mm512_loadu_ps(&x[i + 16]);

        // get lower(0 to 15) and upper(16 to 31) 4 bits
        a8 = _mm_and_si128(a8, mask);
        b8 = _mm_and_si128(_mm_srli_epi16(b8, 4), mask);

        __m512 af = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a8));
        sum = _mm512_fmadd_ps(af, x1, sum);
        __m512 bf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b8));
        sum = _mm512_fmadd_ps(bf, x2, sum);
    }
    float result = _mm512_reduce_add_ps(sum);

    for (size_t i = num32; i < D; ++i) {
        uint8_t cur_code = (y[i / 2] >> (4 * (i % 2 == 0))) & 0b1111;
        result += static_cast<float>(cur_code) * x[i];
    }

    return result;
}

FORCE_INLINE float IP32_fxu6(
    const float* __restrict__ query, const uint8_t* __restrict__ y, size_t D
) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t* o_compact = const_cast<uint8_t*>(y);
    float result = 0;

    __m128i mask6 = _mm_set1_epi8(0b00111111);
    __m128i mask2 = _mm_set1_epi8(0b00110000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);

    for (size_t i = 0; i < D; i += 32) {
        __m128i x = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_compact));
        __m128i y = _mm_cvtsi64_si128(*reinterpret_cast<long long*>(o_compact + 16));

        __m128i vec_0_to_15 = _mm_and_si128(x, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(x, 2), mask2);
        vec_16_to_31 = _mm_or_si128(vec_16_to_31, _mm_and_si128(y, mask4));
        vec_16_to_31 = _mm_or_si128(
            vec_16_to_31, _mm_and_si128(_mm_srli_epi16(_mm_slli_si128(y, 8), 4), mask4)
        );

        __m512 xx = _mm512_loadu_ps(&query[i]);
        __m512 yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_0_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        o_compact += 24;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

FORCE_INLINE float IP64_fxu6(
    const float* __restrict__ query, const uint8_t* __restrict__ y, size_t D
) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t* o_compact = const_cast<uint8_t*>(y);
    float result = 0;

    __m128i mask6 = _mm_set1_epi8(0b00111111);
    __m128i mask2 = _mm_set1_epi8(0b00110000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt1 = _mm_load_si128(reinterpret_cast<__m128i*>(o_compact + 0));
        __m128i cpt2 = _mm_load_si128(reinterpret_cast<__m128i*>(o_compact + 16));
        __m128i cpt3 = _mm_load_si128(reinterpret_cast<__m128i*>(o_compact + 32));

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt1, 2), mask2), _mm_and_si128(cpt3, mask4)
        );
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt2, 2), mask2),
            _mm_and_si128(_mm_srli_epi16(cpt3, 4), mask4)
        );

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        o_compact += 48;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

FORCE_INLINE float IP64_fxu2(
    const float* __restrict__ query, const uint8_t* __restrict__ y, size_t D
) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t* o_compact = const_cast<uint8_t*>(y);
    float result = 0;

    __m128i mask = _mm_set1_epi8(0b00000011);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt = _mm_load_si128(reinterpret_cast<__m128i*>(o_compact));

        __m128i vec_00_to_15 = _mm_and_si128(cpt, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(cpt, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(cpt, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(cpt, 6), mask);

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        o_compact += 16;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

FORCE_INLINE float IP64_fxu7(
    const float* __restrict__ query, const uint8_t* __restrict__ y, size_t D
) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t* o_compact = const_cast<uint8_t*>(y);
    float result = 0;

    __m128i mask6 = _mm_set1_epi8(0b00111111);
    __m128i mask2 = _mm_set1_epi8(0b00110000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);
    __m128i top_mask = _mm_set1_epi8(0b1000000);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_compact + 0));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_compact + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_compact + 32));

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt1, 2), mask2), _mm_and_si128(cpt3, mask4)
        );
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt2, 2), mask2),
            _mm_and_si128(_mm_srli_epi16(cpt3, 4), mask4)
        );
        o_compact += 48;

        int64_t top_bit = *reinterpret_cast<int64_t*>(o_compact);
        o_compact += 8;

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 5, top_bit << 6), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit << 0), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

FORCE_INLINE float IP64_fxu3(
    const float* __restrict__ query, const uint8_t* __restrict__ y, size_t D
) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t* o_compact = const_cast<uint8_t*>(y);
    float result = 0;

    __m128i mask = _mm_set1_epi8(0b11);
    __m128i top_mask = _mm_set1_epi8(0b100);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_compact));
        o_compact += 16;

        int64_t top_bit = *reinterpret_cast<int64_t*>(o_compact);
        o_compact += 8;

        __m128i vec_00_to_15 = _mm_and_si128(cpt, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(cpt, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(cpt, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(cpt, 6), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 5, top_bit >> 4), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum
        );  // I heard that this may cause underclocking on some CPUs.
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

FORCE_INLINE float IP_fxi32(
    const float* __restrict__ x, const int32_t* __restrict__ y, size_t L
) {
    float result = 0;
#if defined(__AVX512F__)
    size_t num16 = L - (L & 0b1111);
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num16; i += 16) {
        __m512 xx = _mm512_loadu_ps(&x[i]);
        __m512 yy = _mm512_cvtepi32_ps(_mm512_loadu_epi32(&y[i]));
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (size_t i = num16; i < L; ++i) {
        result += x[i] * static_cast<float>(y[i]);
    }
#else
    for (size_t i = 0; i < L; ++i) {
        result += x[i] * static_cast<float>(y[i]);
    }
#endif
    return result;
}

/**
 * @brief Sum a vector whose length is a multiple of 16
 *
 * @param x float vec
 * @param L length of vec
 * @return float
 */
inline float sumvec16(const float* __restrict__ x, size_t L) {
    float result = 0;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < L; i += 16) {
        __m512 xx = _mm512_loadu_ps(&x[i]);
        sum = _mm512_add_ps(xx, sum);
    }
    result = _mm512_reduce_add_ps(sum);
#else
    for (size_t i = 0; i < L; i += 4) {
        result += x[i];
        result += x[i + 1];
        result += x[i + 2];
        result += x[i + 3];
    }
#endif
    return result;
}

inline float sumvec(const float* __restrict__ x, size_t L) {
    size_t num16 = L - (L & 0b1111);
    float result = sumvec16(x, num16);
    for (size_t i = num16; i < L; ++i) {
        result += x[i];
    }
    return result;
}

// ==============================================================
// popcount (a.k.a, bitcount)
// ==============================================================
inline size_t popcount(size_t B, const u_int64_t* __restrict__ d) {
    size_t ret = 0;
    for (size_t i = 0; i < B / 64; ++i) {
        ret += __builtin_popcountll((*d));
        ++d;
    }
    return ret;
}

/**
 * @brief Get max abs sum of 4 dim segments of a vec
 *
 * @param qc Target vec
 * @param D Dimension of vec
 * @param vl Lower bound
 * @param vr Upper bound
 */
inline void data_range1(const float* qc, size_t D, float& vl, float& vr) {
    float max_abs = 0;
#if defined(__AVX512F__)
    float PORTABLE_ALIGN64 abs_qc[D];
    for (size_t i = 0; i < D; i += 32) {
        __m512 q1 = _mm512_loadu_ps(&qc[i]);
        __m512 q2 = _mm512_loadu_ps(&qc[i + 16]);
        q1 = _mm512_abs_ps(q1);
        q2 = _mm512_abs_ps(q2);
        _mm512_storeu_ps(&abs_qc[i], q1);
        _mm512_storeu_ps(&abs_qc[i + 16], q2);
    }
    for (size_t i = 0; i < D; i += 4) {
        float tmp = abs_qc[i] + abs_qc[i + 1] + abs_qc[i + 2] + abs_qc[i + 3];
        max_abs = tmp > max_abs ? tmp : max_abs;
    }
#else
    for (size_t i = 0; i < D; i += 4) {
        float tmp = abs(qc[i]) + abs(qc[i + 1]) + abs(qc[i + 2]) + abs(qc[i + 3]);
        max_abs = tmp > max_abs ? tmp : max_abs;
    }
#endif
    vr = max_abs;
    vl = -max_abs;
}

/**
 * @brief get vl & vr for a vector, L must be a multiple of 16, align to 64 bytes
 */
inline void data_range16(const float* __restrict__ q, float& vl, float& vr, size_t L) {
#if defined(__AVX512F__)
    __m512 max_q = _mm512_setzero_ps();
    __m512 min_q = _mm512_setzero_ps();
    for (size_t i = 0; i < L; i += 16) {
        __m512 y = _mm512_load_ps(&q[i]);
        max_q = _mm512_max_ps(y, max_q);
        min_q = _mm512_min_ps(y, min_q);
    }
    vr = _mm512_reduce_max_ps(max_q);
    vl = _mm512_reduce_min_ps(min_q);
#else
    vl = +1e20;
    vr = -1e20;
    for (size_t i = 0; i < L; ++i) {
        float tmp = q[i];
        vl = tmp < vl ? tmp : vl;
        vr = tmp > vr ? tmp : vr;
    }
#endif
}

inline void data_range(const float* __restrict__ q, float& vl, float& vr, size_t L) {
    size_t num16 = L - (L & 0b1111);
    data_range16(q, vl, vr, num16);
    for (size_t i = num16; i < L; ++i) {
        float tmp = q[i];
        vl = tmp < vl ? tmp : vl;
        vr = tmp > vr ? tmp : vr;
    }
}

/*
 * use scalar quantization to quantize a vec
 */
template <typename T, size_t Bits>
inline void scalar_quantize16(
    T* __restrict__ result, const float* __restrict__ vec, float vl, float width, size_t L
) {
    static_assert(sizeof(T) * 8 == Bits);
    float one_over_width = 1.0 / width;
#if defined(__AVX512F__)
    __m512 o = _mm512_set1_ps(one_over_width);
    __m512 v = _mm512_set1_ps(vl);
    __m512 t = _mm512_set1_ps(0.5);
    for (size_t i = 0; i < L; i += 16) {
        __m512 q = _mm512_loadu_ps(&vec[i]);
        q = _mm512_mul_ps(_mm512_sub_ps(q, v), o);
        q = _mm512_add_ps(q, t);

        __m512i c32 = _mm512_cvttps_epi32(q);
        if (Bits == 32) {
            _mm512_storeu_epi32(&result[i], c32);
        } else if (Bits == 16) {
            __m256i c16 = _mm512_cvtepi32_epi16(c32);
            _mm256_storeu_epi32(&result[i], c16);
        } else if (Bits == 8) {
            __m128i c8 = _mm256_cvtepi16_epi8(_mm512_cvtepi32_epi16(c32));
            _mm_storeu_epi32(&result[i], c8);
        } else {
            std::cerr << "Bad scalar quantization parameters\n";
            abort();
        }
    }
#else
    for (size_t i = 0; i < L; i += 4) {
        result[i] = static_cast<T>((vec[i] - vl) * one_over_width + 0.5);
        result[i + 1] = static_cast<T>((vec[i + 1] - vl) * one_over_width + 0.5);
        result[i + 2] = static_cast<T>((vec[i + 2] - vl) * one_over_width + 0.5);
        result[i + 3] = static_cast<T>((vec[i + 3] - vl) * one_over_width + 0.5);
    }
#endif
}

template <typename T, size_t Bits>
inline void scalar_quantize(
    T* __restrict__ result, const float* __restrict__ vec, float vl, float width, size_t L
) {
    size_t num16 = L - (L & 0b1111);
    scalar_quantize16<T, Bits>(result, vec, vl, width, num16);

    float one_over_width = 1.0 / width;
    for (size_t i = num16; i < L; ++i) {
        result[i] = static_cast<T>((vec[i] - vl) * one_over_width + 0.5);
    }
}

void high_acc_quantize16(
    int16_t* __restrict__ result, const float* __restrict__ q, float& width, size_t D
) {
    // quantize it into 14-bit SIGNED intger such that
    // the sum of 4 does not cause overflow in int16
    constexpr int BQ = 14;
    // find the maximum absolute value in the vector q;
    float vl, vr;
    data_range16(q, vl, vr, D);
    float vmax = std::max(std::abs(vl), std::abs(vr));

    width = vmax / ((1 << (BQ - 1)) - 1);
    for (size_t i = 0; i < D; i++) {
        float tmp = q[i] / width;
        result[i] = (int16_t)(tmp + 0.5 - (tmp < 0));
    }
}

/* sub 2 vec and get summary of new vec, L % 16 = 0 */
inline float minus_sum16(
    const float* __restrict__ q,
    const float* __restrict__ c,
    float* __restrict__ qc,
    size_t L
) {
#if defined(__AVX512F__)
    const float* x = q;
    const float* y = c;
    float* z = qc;
    auto sum = _mm512_setzero_ps();
    for (size_t j = 0; j < L; j += 16) {
        auto xx = _mm512_loadu_ps(x);
        auto yy = _mm512_loadu_ps(y);
        auto zz = _mm512_sub_ps(xx, yy);
        sum = _mm512_add_ps(sum, zz);
        _mm512_storeu_ps(z, zz);
        x += 16;
        y += 16;
        z += 16;
    }
    return _mm512_reduce_add_ps(sum);
#else
    float sumqc = 0;
    for (size_t i = 0; i < L; i += 4) {
        qc[i] = q[i] - c[i];
        qc[i + 1] = q[i + 1] - c[i + 1];
        qc[i + 2] = q[i + 2] - c[i + 2];
        qc[i + 3] = q[i + 3] - c[i + 3];
        sumqc += qc[i] + qc[i + 1] + qc[i + 2] + qc[i + 3];
    }
    return sumqc;
#endif
}

inline float normalize_query16(
    float* unit_q,
    const float* __restrict__ q,
    const float* __restrict__ c,
    float norm,
    size_t D
) {
    constexpr float eps = 1e-5;

    if (norm > eps) {
        auto sum = _mm512_setzero_ps();
        auto fac = _mm512_set1_ps(1.0 / norm);
        for (size_t i = 0; i < D; i += 16) {
            auto qq = _mm512_loadu_ps(q);
            auto cc = _mm512_loadu_ps(c);
            auto uu = _mm512_mul_ps(_mm512_sub_ps(qq, cc), fac);
            sum = _mm512_add_ps(sum, uu);
            _mm512_storeu_ps(unit_q, uu);

            unit_q += 16;
            q += 16;
            c += 16;
        }
        return _mm512_reduce_add_ps(sum);
    } else {  // in case that q_r-c is a zero vector
        float value = 1.0 / std::sqrt((float)D);
        std::fill(unit_q, unit_q + D, value);
        return D * value;
    }
}

inline float minus_sum(
    const float* __restrict__ q,
    const float* __restrict__ c,
    float* __restrict__ qc,
    size_t L
) {
    size_t num16 = L - (L & 0b1111);
    float sumqc = minus_sum16(q, c, qc, num16);

    for (size_t i = num16; i < L; ++i) {
        qc[i] = q[i] - c[i];
        sumqc += qc[i];
    }
    return sumqc;
}