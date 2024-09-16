#pragma once

#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <vector>

#include "defines.hpp"
#include "index/Quantizer.hpp"
#include "utils/memory.hpp"
#include "utils/tools.hpp"

class Cluster {
   private:
    size_t Num;                     // Num of vectors in this cluster
    size_t ITER;                    // Num of iter for FastScan
    size_t REMAIN;                  // Num of remained vectors after blocks
    uint8_t* SHORT_DATA = nullptr;  // RaBitQ code and factors
    uint8_t* LONG_CODE = nullptr;   // ExRaBitQ code
    ExFactor* EX_FACTOR = nullptr;  // ExRaBitQ factor
    PID* IDs = nullptr;             // PID of vectors

   public:
    explicit Cluster(size_t, uint8_t*, uint8_t*, ExFactor*, PID*);
    Cluster(const Cluster& other);
    Cluster(Cluster&& other) noexcept;
    ~Cluster() {}

    /**
     * @brief Return pointer to first block
     */
    uint8_t* first_block() const { return this->SHORT_DATA; }

    /**
     * @brief Return long code for i-th vector in this cluster
     */
    uint8_t* long_code(size_t i, const DataQuantizer& DQ) const {
        return this->LONG_CODE + DQ.long_code_length() * i;
    }

    /**
     * @brief Return long factor of ith
     */
    ExFactor* ex_factor(size_t i) const { return this->EX_FACTOR + i; }

    /**
     * @brief Return pointer to ids
     */
    PID* ids() const { return this->IDs; }

    size_t num() const { return Num; }

    size_t iter() const { return ITER; }

    size_t remain() const { return REMAIN; }
};

/**
 * @brief Construct a new Cluster:: Cluster object
 * Data in the cluster are mapped to large arrays in memory
 *
 * @param num number of vectors
 * @param short_data blocks of 1-bit codes and corresponding factors
 * @param long_code long code for re-ranking
 * @param ex_factor factors for re-ranking
 * @param ids id for vectors in the cluster
 */
Cluster::Cluster(
    size_t num, uint8_t* short_data, uint8_t* long_code, ExFactor* ex_factor, PID* ids
)
    : Num(num)
    , ITER(num / FAST_SIZE)
    , REMAIN(num - ITER * FAST_SIZE)
    , SHORT_DATA(short_data)
    , LONG_CODE(long_code)
    , EX_FACTOR(ex_factor)
    , IDs(ids) {}

/**
 * @brief Copy contructor
 */
Cluster::Cluster(const Cluster& other)
    : Num(other.Num)
    , ITER(other.ITER)
    , REMAIN(other.REMAIN)
    , SHORT_DATA(other.SHORT_DATA)
    , LONG_CODE(other.LONG_CODE)
    , EX_FACTOR(other.EX_FACTOR)
    , IDs(other.IDs) {}

/**
 * @brief Move contructor
 */
Cluster::Cluster(Cluster&& other) noexcept
    : Num(other.Num)
    , ITER(other.ITER)
    , REMAIN(other.REMAIN)
    , SHORT_DATA(other.SHORT_DATA)
    , LONG_CODE(other.LONG_CODE)
    , EX_FACTOR(other.EX_FACTOR)
    , IDs(other.IDs) {}
