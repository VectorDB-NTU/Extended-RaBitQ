#pragma once

#include <stdint.h>

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "defines.hpp"
#include "third/hnswlib/hnswlib.h"
#include "utils/memory.hpp"
#include "utils/space.hpp"

class Initializer {
   protected:
    size_t D;
    size_t K;

   public:
    explicit Initializer(size_t d, size_t k) : D(d), K(k) {}
    virtual ~Initializer() {}
    virtual float* centroid(PID) const = 0;
    virtual void add_vectors(const float*) = 0;
    virtual void centroids_distances(const float*, size_t, std::vector<Candidate>&)
        const = 0;
    virtual void load(std::ifstream&, const char*) = 0;
    virtual void save(std::ofstream&, const char*) const = 0;
};

class FlatInitializer : public Initializer {
   private:
    float* Centroids = nullptr;
    float (*dist_func)(const float* __restrict__, const float* __restrict__, size_t);
    size_t data_bytes() const { return sizeof(float) * K * D; }

   public:
    explicit FlatInitializer(size_t d, size_t k) : Initializer(d, k) {
        this->Centroids = memory::align_mm<64, float>(data_bytes());
        if (d % 16 == 0) {
            dist_func = L2Sqr16;
        } else {
            dist_func = L2Sqr;
        }
    }

    ~FlatInitializer() { std::free(Centroids); }

    float* centroid(PID id) const override { return this->Centroids + id * D; }

    void add_vectors(const float* cent) override {
        std::memcpy(this->Centroids, cent, sizeof(float) * K * D);
    }

    void centroids_distances(
        const float* query, size_t nprobe, std::vector<Candidate>& candidates
    ) const override {
        std::vector<Candidate> centroid_dist(this->K);
        for (PID i = 0; i < K; ++i) {
            centroid_dist[i].id = i;
            centroid_dist[i].distance = dist_func(query, centroid(i), D);
        }
        std::partial_sort(
            centroid_dist.begin(), centroid_dist.begin() + nprobe, centroid_dist.end()
        );

        std::memcpy(candidates.data(), centroid_dist.data(), sizeof(Candidate) * nprobe);
    }

    void save(std::ofstream& output, const char*) const override {
        output.write((char*)Centroids, data_bytes());
    }

    void load(std::ifstream& input, const char*) override {
        input.read((char*)Centroids, data_bytes());
    }
};

class HNSWInitializer : public Initializer {
   private:
    static constexpr int M = 16;
    static constexpr int ef_construction = 400;
    hnswlib::HierarchicalNSW<float>* alg_hnsw = nullptr;
    hnswlib::L2Space space;

   public:
    explicit HNSWInitializer(size_t d, size_t k) : Initializer(d, k), space(d) {
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, K, M, ef_construction);
    }

    void add_vectors(const float* cent) override {
        std::cout << "Inserting vectors into hnsw...\n";
        for (size_t i = 0; i < K; ++i) {
            alg_hnsw->addPoint(cent + i * D, i);
        }
        std::cout << "Inserted vectors into hnsw...\n";
    }

    float* centroid(PID id) const override {
        return (float*)alg_hnsw->getDataByInternalId(id);
    }

    void centroids_distances(
        const float* query, size_t nprobe, std::vector<Candidate>& candidates
    ) const override {
        alg_hnsw->setEf(std::max(768ul, 2 * nprobe));
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw->searchKnn(query, nprobe);

        for (size_t i = 0; i < nprobe; ++i) {
            candidates[i].distance = result.top().first;
            candidates[i].id = result.top().second;
            result.pop();
        }
    }

    void save(std::ofstream&, const char* filename) const override {
        std::string hnsw(filename);
        hnsw += ".hnsw";
        alg_hnsw->saveIndex(hnsw);
    }

    void load(std::ifstream&, const char* filename) override {
        std::string hnsw(filename);
        hnsw += ".hnsw";
        alg_hnsw->loadIndex(hnsw, &space, K);
    }

    ~HNSWInitializer() { delete alg_hnsw; }
};