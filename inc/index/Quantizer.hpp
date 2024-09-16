#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <queue>
#include <vector>

#include "defines.hpp"
#include "index/Rotator.hpp"
#include "index/fastscan/pack_codes.hpp"
#include "utils/space.hpp"
#include "utils/tools.hpp"

/**
 * @brief Quantizer of ExRaBitQ. Including 2 quantization codes
 *
 * @tparam B number of bits
 */
class DataQuantizer {
   private:
    size_t DIM;                // Dimension of data objects
    size_t D;                  // Packed DIM (multiple of 64)
    size_t EX_BITS;            // Number of bits for ExRaBitQ
    size_t SHORT_CODE_LENGTH;  // Number of uint8 to store 1bit code for a vec
    size_t LONG_CODE_LENGTH;   // Number of uint8 to store EX_BITS code for a vec
    double FAC_NORM;
    double FAC_ERR;
#if defined(HIGH_ACC_FAST_SCAN)
    static constexpr size_t NUM_SHORT_FACTORS = 1;
#else
    static constexpr size_t NUM_SHORT_FACTORS = 4;
#endif

    void pack_binary(const IntRowMat&, uint64_t*, size_t) const;

    void
    rabitq_factor(const float*, const float*, const std::vector<PID>&, const IntRowMat&, const FloatRowMat&, float*, float*, float*, float*)
        const;

    void fast_quantize(const float*, uint8_t*, float&) const;

    void rabitq_codes(const IntRowMat&, uint8_t*) const;

    void
    exrabitq_codes(const IntRowMat&, const FloatRowMat&, uint8_t*, ExFactor*, const float*)
        const;

    void
    data_transformation(const float*, const float*, const std::vector<PID>&, const Rotator&, float*, FloatRowMat&, IntRowMat&)
        const;

    void store_compacted_code(uint8_t*, uint8_t*) const;

   public:
    explicit DataQuantizer(size_t dim, size_t b)
        : DIM(dim)
        , D(rd_up_to_multiple_of(dim, 64))
        , EX_BITS(b)
        , SHORT_CODE_LENGTH(D / 8)
        , LONG_CODE_LENGTH(D * EX_BITS / 8)  // TODO: handle more cases of EX_BITS
        , FAC_NORM(1 / std::sqrt((double)D))
        , FAC_ERR(2.0 / std::sqrt((double)(D - 1))) {}

    explicit DataQuantizer() {}

    constexpr DataQuantizer& operator=(const DataQuantizer& other) {
        this->DIM = other.DIM;
        this->D = other.D;
        this->EX_BITS = other.EX_BITS;
        this->SHORT_CODE_LENGTH = other.SHORT_CODE_LENGTH;
        this->LONG_CODE_LENGTH = other.LONG_CODE_LENGTH;
        this->FAC_NORM = other.FAC_NORM;
        this->FAC_ERR = other.FAC_ERR;
        return *this;
    }

    size_t short_code_length() const { return SHORT_CODE_LENGTH; }

    size_t long_code_length() const { return LONG_CODE_LENGTH; }

    size_t block_bytes() const {
        return SHORT_CODE_LENGTH * FAST_SIZE * sizeof(uint8_t) +
               sizeof(float) * num_short_factors() * FAST_SIZE;
    }

    size_t num_blocks(size_t num) const { return div_rd_up(num, FAST_SIZE); };

    static constexpr size_t num_short_factors() { return NUM_SHORT_FACTORS; }

    void
    quantize(const float*, const float*, const std::vector<PID>&, const Rotator&, uint8_t*, uint8_t*, ExFactor*, float*)
        const;

    /**
     * @brief Get pointer of factors for current block
     * Each block (stored in uint8_t*) contains the 1st bit of quantization code
     * which is used for FastScan and computing approximate distances by SIMD instructions.
     *
     * @param block Pointer to current block
     * @return float*
     */
    float* block_factor(uint8_t* block) const {
        return reinterpret_cast<float*>(&block[D << 2]);  // FAST_SIZE * D / 8
    }
    // sqr of distance to cluster centroid
    float* factor_x2(float* block_fac) const { return block_fac; }

    // 1/ip * 2x
    float* factor_ip(float* block_fac) const { return &block_fac[FAST_SIZE]; }

    // number of 1s in the D-bit string
    float* factor_sumxb(float* block_fac) const { return &block_fac[FAST_SIZE * 2]; }

    // error factor of RaBitQ
    float* factor_err(float* block_fac) const { return &block_fac[FAST_SIZE * 3]; }

    // Get pointer to next block
    uint8_t* next_block(float* block_fac) const {
        return reinterpret_cast<uint8_t*>(&block_fac[FAST_SIZE * num_short_factors()]);
    }
};

/**
 * @brief Compute norm and inner product for ExRaBitQ
 * @note  Currently assume all number in o_prime are positive. For negative
 * values, the quantization code needs to be flipped.
 *
 * @param o_prime   Rotated data vector, length of D
 * @param o_bar     Quantization code of ExRaBitQ, length of D
 * @param norm      L2 norm of quantized vec
 * @param ip        <oex_bar, o_bar>, ~ 1
 */
void DataQuantizer::fast_quantize(const float* o_prime, uint8_t* code, float& ip_norm)
    const {
    constexpr double eps = 1e-5;
    constexpr int n_enum = 10;
    double max_o = -1;

    for (size_t i = 0; i < D; i++)
        if (o_prime[i] > max_o)
            max_o = o_prime[i];
    double t_start = (double)(((1 << EX_BITS) - 1) / 3) / max_o;
    double t_end = (double)(((1 << EX_BITS) - 1) + n_enum) / max_o;

    int cur_o_bar[D];
    double sqr_denominator = D * 0.25;
    double numerator = 0;

    for (size_t i = 0; i < D; ++i) {
        cur_o_bar[i] = int((double)t_start * o_prime[i] + eps);
        sqr_denominator += cur_o_bar[i] * cur_o_bar[i] + cur_o_bar[i];
        numerator += (cur_o_bar[i] + 0.5) * o_prime[i];
    }

    std::priority_queue<
        std::pair<double, size_t>,
        std::vector<std::pair<double, size_t>>,
        std::greater<std::pair<double, size_t>>>
        next_t;

    for (size_t i = 0; i < D; ++i) {
        next_t.emplace(std::make_pair((double)(cur_o_bar[i] + 1) / o_prime[i], i));
    }

    double max_ip = 0;
    double t = 0;

    size_t cnt = 0;
    while (next_t.empty() == false) {
        double cur_t = next_t.top().first;
        size_t update_id = next_t.top().second;
        ++cnt;
        next_t.pop();

        cur_o_bar[update_id]++;
        int update_o_bar = cur_o_bar[update_id];
        sqr_denominator += 2 * update_o_bar;
        numerator += o_prime[update_id];

        double cur_ip = numerator / std::sqrt(sqr_denominator);
        if (cur_ip > max_ip) {
            max_ip = cur_ip;
            t = cur_t;
        }

        if (update_o_bar < (1 << EX_BITS) - 1) {
            double t_next = (double)(update_o_bar + 1) / o_prime[update_id];
            if (t_next < t_end)
                next_t.emplace(std::make_pair(t_next, update_id));
        }
    }

    sqr_denominator = D * 0.25;
    numerator = 0;
    int32_t o_bar[D];
    for (size_t i = 0; i < D; i++) {
        o_bar[i] = int((double)t * o_prime[i] + eps);
        if (o_bar[i] >= (1 << EX_BITS))
            o_bar[i] = (1 << EX_BITS) - 1;
        sqr_denominator += (int)o_bar[i] * o_bar[i] + o_bar[i];
        numerator += (o_bar[i] + 0.5) * o_prime[i];
    }
    // double norm = std::sqrt(sqr_denominator);
    // double ip = numerator / norm;
    ip_norm = 1 / numerator;  // 1/(ip*norm)
    if (!std::isfinite(ip_norm)) {
        ip_norm = 1.f;
    }

    /* Store code as uint8 */
    for (size_t i = 0; i < D; ++i) {
        code[i] = static_cast<uint8_t>(o_bar[i]);
    }
}

/**
 * @brief Quantize data vectors of IDs and store related data
 *
 * @param data          Raw data vectors (N * DIM)
 * @param centroid      Centroid data vector (1 * DIM)
 * @param IDs           IDs of vectors to quantize
 * @param rotator            Rotator of data vector
 * @param short_data    Packed quantization code and factors for RaBitQ used for FastScan
 * @param long_code     Quantization code for ExRaBitQ
 * @param ex_factor     Factor for re-ranking
 * @param rotated_c     Rotated centroid, (1 * D)
 */
void DataQuantizer::quantize(
    const float* data,
    const float* centroid,
    const std::vector<PID>& IDs,
    const Rotator& rotator,
    uint8_t* short_data,
    uint8_t* long_code,
    ExFactor* ex_factor,
    float* rotated_c
) const {
    size_t num_points = IDs.size();  // Num of point in this cluster

    FloatRowMat XP_norm;
    IntRowMat bin_XP;

    data_transformation(data, centroid, IDs, rotator, rotated_c, XP_norm, bin_XP);

    size_t total_blocks = num_blocks(num_points);
    uint8_t* all_short_codes = new uint8_t[short_code_length() * FAST_SIZE * total_blocks];
    float* all_factor_x2 = new float[FAST_SIZE * total_blocks];
    float* all_factor_ip = new float[FAST_SIZE * total_blocks];
    float* all_factor_sumxb = new float[FAST_SIZE * total_blocks];
    float* all_factor_err = new float[FAST_SIZE * total_blocks];

    /* quantization code of RaBitQ */
    rabitq_codes(bin_XP, all_short_codes);

    /* pre-computed factor */
    rabitq_factor(
        data,
        centroid,
        IDs,
        bin_XP,
        XP_norm,
        all_factor_x2,
        all_factor_ip,
        all_factor_sumxb,
        all_factor_err
    );

    /* quantization code of ExRaBitQ */
    exrabitq_codes(bin_XP, XP_norm, long_code, ex_factor, all_factor_x2);

    /* Save short codes and factor block by block */
    uint8_t* block = short_data;
    for (size_t i = 0; i < total_blocks; ++i) {
        // copy codes
        std::memcpy(
            block,
            &all_short_codes[i * short_code_length() * FAST_SIZE],
            sizeof(uint8_t) * short_code_length() * FAST_SIZE
        );

#if defined(HIGH_ACC_FAST_SCAN)
        float* block_fac = block_factor(block);
        float* cur_x2 = factor_x2(block_fac);
        std::memcpy(cur_x2, &all_factor_x2[i * FAST_SIZE], sizeof(float) * FAST_SIZE);
#else
        // copy factors
        float* block_fac = block_factor(block);
        float* cur_x2 = factor_x2(block_fac);
        float* cur_ip = factor_ip(block_fac);
        float* cur_sumxb = factor_sumxb(block_fac);
        float* cur_err = factor_err(block_fac);
        std::memcpy(cur_x2, &all_factor_x2[i * FAST_SIZE], sizeof(float) * FAST_SIZE);
        std::memcpy(cur_ip, &all_factor_ip[i * FAST_SIZE], sizeof(float) * FAST_SIZE);
        std::memcpy(cur_sumxb, &all_factor_sumxb[i * FAST_SIZE], sizeof(float) * FAST_SIZE);
        std::memcpy(cur_err, &all_factor_err[i * FAST_SIZE], sizeof(float) * FAST_SIZE);
#endif

        block = next_block(block_fac);
    }

    delete[] all_short_codes;
    delete[] all_factor_x2;
    delete[] all_factor_ip;
    delete[] all_factor_sumxb;
    delete[] all_factor_err;
}

/**
 * @brief Normalize & rotate data for quantization
 */
void DataQuantizer::data_transformation(
    const float* data,
    const float* centroid,
    const std::vector<PID>& IDs,
    const Rotator& rotator,
    float* rotated_c,
    FloatRowMat& XP_norm,
    IntRowMat& bin_XP
) const {
    /* Assesrt correct size */
    assert(rotator.size() == D);

    size_t num_points = IDs.size();  // Num of point in this cluster

    FloatRowMat X_pad(num_points, this->D);  // padded data points mat
    FloatRowMat C_pad(1, this->D);           // padded centroid mat
    X_pad.setZero();
    C_pad.setZero();

    /* Copy data */
    size_t copy_size = this->DIM * sizeof(float);  // Num of bytes for each data vector
    for (size_t i = 0; i < num_points; ++i) {
        const float* cur_data = data + this->DIM * IDs[i];
        std::memcpy(&X_pad(i, 0), cur_data, copy_size);
    }
    std::memcpy(C_pad.data(), centroid, copy_size);

    /* Rotate Data */
    FloatRowMat XP(num_points, this->D);
    FloatRowMat CP(1, this->D);
    rotator.rotate(X_pad, XP);
    rotator.rotate(C_pad, CP);
    for (int i = 0; i < XP.rows(); i++) {
        XP.row(i) = XP.row(i) - CP;  // residual
    }
    std::memcpy(rotated_c, CP.data(),
                sizeof(float) * D);       // save rotated centroid
    XP_norm = XP.rowwise().normalized();  // normalized rotated data

    /* Binary representation */
    bin_XP = IntRowMat(num_points, this->D);
    for (uint32_t i = 0; i < num_points; ++i) {
        for (uint32_t j = 0; j < this->D; ++j) {
            bin_XP(i, j) = (XP(i, j) > 0);
        }
    }
}

/**
 * @brief Change 0/1 mat to uint64_t
 *
 * @param bin_XP    binary mat (bit string)
 * @param binary    uint64 representation of bin_XP
 * @param num_points
 */
void DataQuantizer::pack_binary(
    const IntRowMat& bin_XP, uint64_t* binary, size_t num_points
) const {
    for (size_t row = 0; row < num_points; ++row) {
        for (size_t col = 0; col < this->D; col += 64) {
            uint64_t cur = 0;
            for (size_t i = 0; i < 64; ++i) {
                cur |= (static_cast<uint64_t>(bin_XP(row, col + i)) << (63 - i));
            }
            *binary = cur;
            ++binary;
        }
    }
}

void DataQuantizer::rabitq_factor(
    const float* data,
    const float* centroid,
    const std::vector<PID>& IDs,
    const IntRowMat& bin_XP,
    const FloatRowMat& XP_norm,
    float* fac_x2,
    float* fac_ip,
    float* fac_sumxb,
    float* fac_err
) const {
    size_t num_points = IDs.size();

    /* Signed quantized vectors */
    FloatRowMat Ones(num_points, this->D);
    Ones.setOnes();
    FloatRowMat coded_vec = (2 * bin_XP.cast<float>() - Ones) * FAC_NORM;

    /* X0: <o,o_bar> */
    FloatRowMat X0 = (XP_norm.array() * coded_vec.array()).rowwise().sum();

    for (size_t i = 0; i < num_points; ++i) {
        // distance 2 centroid
        const float* cur_data = data + this->DIM * IDs[i];
        fac_x2[i] = L2Sqr(cur_data, centroid, this->DIM);
        float dist2c = std::sqrt((double)fac_x2[i]);  // x
#if defined(HIGH_ACC_FAST_SCAN)
        fac_x2[i] = dist2c;
#endif

        // 1/(<o,o_bar> * sqrt(D))
        float o_obar = X0(i, 0);
        if (!std::isfinite(o_obar)) {  // handle bad value
            o_obar = 0.8;
        }

        // fac ip
        double ip = 0;
        for (size_t j = 0; j < D; ++j) {
            ip += std::abs(0.5 * XP_norm(i, j));
        }
        fac_ip[i] = 1.0 / ip * 2 * dist2c;

        // fac sumxb
        int sumxb = bin_XP.row(i).sum();
        fac_sumxb[i] = static_cast<float>(sumxb);

        // error factor
        fac_err[i] =
            std::sqrt((1.0 - o_obar * o_obar) / (o_obar * o_obar)) * FAC_ERR * 2 * dist2c;
    }
}

/**
 * @brief Get quantization code of RaBitQ
 *
 * @param bin_XP        binary representation of matrix XP
 * @param num_points    number of vecs
 * @param packed_code   packed code for FastScan
 */
void DataQuantizer::rabitq_codes(const IntRowMat& bin_XP, uint8_t* packed_code) const {
    size_t num_points = bin_XP.rows();
    /* change bin_XP to uint64 */
    uint64_t* binary = new uint64_t[num_points * (this->D / 64)];
    pack_binary(bin_XP, binary, num_points);

    pack_codes(this->D, binary, num_points, packed_code);

    delete[] binary;
}

void DataQuantizer::exrabitq_codes(
    const IntRowMat& bin_XP,
    const FloatRowMat& XP_norm,
    uint8_t* long_code,
    ExFactor* ex_factor,
    const float* fac_x2
) const {
    size_t num_points = bin_XP.rows();
    int32_t mask = (1 << EX_BITS) - 1;
    FloatRowMat abs_mat = XP_norm.array().abs();
    uint8_t PORTABLE_ALIGN64 tmp_code[D];

    // get long codes
    for (size_t i = 0; i < num_points; ++i) {
        float ipnorm;
        fast_quantize(&abs_mat(i, 0), tmp_code, ipnorm);
#if defined(HIGH_ACC_FAST_SCAN)
        ex_factor[i].xipnorm = ipnorm * 2 * fac_x2[i];
#else
        ex_factor[i].xipnorm = ipnorm * 2 * std::sqrt((double)fac_x2[i]);
#endif

        // revert codes for negative dims
        for (size_t j = 0; j < D; ++j) {
            if (bin_XP(i, j) == 0) {
                uint8_t tmp = tmp_code[j];
                tmp_code[j] = (~tmp) & mask;
            }
        }

        store_compacted_code(tmp_code, long_code + i * long_code_length());
    }
}

void DataQuantizer::store_compacted_code(uint8_t* o_raw, uint8_t* o_compact) const {
    if (EX_BITS == 8) {
        std::memcpy(o_compact, o_raw, sizeof(uint8_t) * D);
    } else if (EX_BITS == 4) {
        for (size_t j = 0; j < D; j += 32) {
            __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw));
            __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 16));
            vec_16_to_31 = _mm_slli_epi16(vec_16_to_31, 4);

            __m128i compact = _mm_or_si128(vec_00_to_15, vec_16_to_31);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);

            o_raw += 32;
            o_compact += 16;
        }
    } else if (EX_BITS == 6) {
        __m128i mask2 = _mm_set1_epi8(0b11000000);
        __m128i mask4 = _mm_set1_epi8(0b00001111);
        for (size_t d = 0; d < D; d += 64) {
            __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw));
            __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 16));
            __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 32));
            __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 48));

            __m128i compact = _mm_or_si128(
                vec_00_to_15, _mm_and_si128(_mm_slli_epi16(vec_32_to_47, 2), mask2)
            );
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 0), compact);

            compact = _mm_or_si128(
                vec_16_to_31, _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 2), mask2)
            );
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 16), compact);

            compact = _mm_or_si128(
                _mm_and_si128(vec_32_to_47, mask4),
                _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask4), 4)
            );
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 32), compact);

            o_raw += 64;
            o_compact += 48;
        }
    } else if (EX_BITS == 2) {
        __m128i mask = _mm_set1_epi8(0b00000011);
        for (size_t d = 0; d < D; d += 64) {
            __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw));
            __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 16));
            __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 32));
            __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 48));

            vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
            vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 2);
            vec_32_to_47 = _mm_slli_epi16(_mm_and_si128(vec_32_to_47, mask), 4);
            vec_48_to_63 = _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask), 6);

            __m128i compact = _mm_or_si128(
                _mm_or_si128(vec_00_to_15, vec_16_to_31),
                _mm_or_si128(vec_32_to_47, vec_48_to_63)
            );

            _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);

            o_raw += 64;
            o_compact += 16;
        }
    } else if (EX_BITS == 3) {
        __m128i mask = _mm_set1_epi8(0b11);
        for (size_t d = 0; d < D; d += 64) {
            __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw));
            __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 16));
            __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 32));
            __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 48));

            vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
            vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 2);
            vec_32_to_47 = _mm_slli_epi16(_mm_and_si128(vec_32_to_47, mask), 4);
            vec_48_to_63 = _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask), 6);

            __m128i compact = _mm_or_si128(
                _mm_or_si128(vec_00_to_15, vec_16_to_31),
                _mm_or_si128(vec_32_to_47, vec_48_to_63)
            );

            _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);
            o_compact += 16;

            int64_t top_bit = 0;
            int64_t top_mask = 0x0101010101010101;
            for (size_t i = 0; i < 64; i += 8) {
                int64_t cur_codes = *reinterpret_cast<int64_t*>(o_raw + i);
                top_bit |= ((cur_codes >> 2) & top_mask) << (i / 8);
            }
            std::memcpy(o_compact, &top_bit, sizeof(int64_t));

            o_raw += 64;
            o_compact += 8;
        }
    } else if (EX_BITS == 7) {
        __m128i mask2 = _mm_set1_epi8(0b11000000);
        __m128i mask4 = _mm_set1_epi8(0b00001111);
        __m128i mask6 = _mm_set1_epi8(0b00111111);
        for (size_t d = 0; d < D; d += 64) {
            __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw));
            __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 16));
            __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 32));
            __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<__m128i*>(o_raw + 48));

            __m128i compact = _mm_or_si128(
                _mm_and_si128(vec_00_to_15, mask6),
                _mm_and_si128(_mm_slli_epi16(vec_32_to_47, 2), mask2)
            );
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 0), compact);

            compact = _mm_or_si128(
                _mm_and_si128(vec_16_to_31, mask6),
                _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 2), mask2)
            );
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 16), compact);

            compact = _mm_or_si128(
                _mm_and_si128(vec_32_to_47, mask4),
                _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask4), 4)
            );
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 32), compact);
            o_compact += 48;

            int64_t top_bit = 0;
            int64_t top_mask = 0x0101010101010101;
            for (size_t i = 0; i < 64; i += 8) {
                int64_t cur_codes = *reinterpret_cast<int64_t*>(o_raw + i);
                top_bit |= ((cur_codes >> 6) & top_mask) << (i / 8);
            }
            std::memcpy(o_compact, &top_bit, sizeof(int64_t));

            o_compact += 8;
            o_raw += 64;
        }
    }
}