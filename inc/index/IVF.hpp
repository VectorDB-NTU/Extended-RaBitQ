#pragma once

#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "defines.hpp"
#include "index/Cluster.hpp"
#include "index/HASearcher.hpp"
#include "index/Searcher.hpp"
#include "index/Initializer.hpp"
#include "index/Pool.hpp"
#include "index/Quantizer.hpp"
#include "index/Rotator.hpp"
#include "index/fastscan/FastScan.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"
#include "utils/memory.hpp"
#include "utils/space.hpp"

class IVF {
   private:
    Initializer* initer = nullptr;
    uint8_t* SHORT_DATA = nullptr;    // RaBitQ code and factors
    uint8_t* LONG_CODE = nullptr;     // ExRaBitQ code
    ExFactor* EX_FACTOR = nullptr;    // ExRaBitQ factor
    PID* IDs = nullptr;               // PID of vectors
    size_t N;                         // num of data points
    size_t DIM;                       // dimension of data points
    size_t D;                         // padded dimension
    size_t K;                         // num of centroids
    size_t EX_BITS;                   // totalbits = EX_BITS+1
    DataQuantizer DQ;                 // Data quantizer
    Rotator Rota;                     // Data Rotator
    std::vector<Cluster> ClusterLst;  // List of clusters in IVF

    void
    quantize_cluster(Cluster&, const std::vector<PID>&, const float*, const float*, float*);

    size_t exfactor_bytes() const { return sizeof(ExFactor) * N; }

    size_t ids_bytes() const { return sizeof(PID) * N; }

    size_t longcode_bytes() const { return sizeof(uint8_t) * DQ.long_code_length() * N; }

    size_t shortdata_bytes(const std::vector<size_t>& cluster_sizes) const {
        assert(cluster_sizes.size() == K);  // num of clusters
        size_t total_blocks = 0;
        for (auto s : cluster_sizes) {
            total_blocks += DQ.num_blocks(s);
        }
        return total_blocks * DQ.block_bytes();
    }

    void allocate_memory(const std::vector<size_t>&);

    void init_clusters(const std::vector<size_t>&);

    void free_memory() {
        delete initer;
        std::free(SHORT_DATA);
        std::free(LONG_CODE);
        std::free(EX_FACTOR);
        std::free(IDs);
    }

   public:
    explicit IVF() {}
    explicit IVF(size_t, size_t, size_t, size_t);

    ~IVF();

    void construct(const float*, const float*, const PID*);

    void save(const char*) const;

    void load(const char*);

    void search(const float*, const float*, size_t, size_t, PID*) const;

    size_t padded_dim() { return this->D; }

    Rotator& rotator() { return this->Rota; }

    size_t k() const { return K; }
};

IVF::IVF(size_t n, size_t dim, size_t k, size_t b)
    : N(n)
    , DIM(dim)
    , D(rd_up_to_multiple_of(dim, 64))
    , K(k)
    , EX_BITS(b - 1)
    , DQ(DIM, EX_BITS)
    , Rota(DIM) {
    assert(
        EX_BITS == 8 || EX_BITS == 4 || EX_BITS == 6 || EX_BITS == 2 || EX_BITS == 3 ||
        EX_BITS == 7
    );
}

IVF::~IVF() { free_memory(); }

/**
 * @brief Construct clusters in IVF
 *
 * @param data Data objects (N*DIM)
 * @param centroids Centroid vectors (K*DIM)
 * @param clustter_ids Cluster ID for each data objects
 */
void IVF::construct(const float* data, const float* centroids, const PID* cluster_ids) {
    std::cout << "Start IVF construction...\n";

    /* Get data ids belonging to each cluster */
    std::cout << "\tLoading clustering information...\n";
    std::vector<size_t> counts(K, 0);
    std::vector<std::vector<PID>> IDLists(K);
    for (size_t i = 0; i < N; ++i) {
        PID cid = cluster_ids[i];
        if (cid > K) {
            std::cerr << "Bad cluster id\n";
            abort();
        }
        IDLists[cid].push_back((PID)i);
        counts[cid] += 1;
    }

    /* Allocate memory */
    allocate_memory(counts);

    /* Init each cluster */
    init_clusters(counts);

    float* rotated_centroids = new float[K * D];

    /* Quantize each cluster */
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < K; ++i) {
        const float* cur_centroid = centroids + i * DIM;
        float* cur_rotated_c = rotated_centroids + i * D;
        Cluster& cp = ClusterLst[i];
        quantize_cluster(cp, IDLists[i], data, cur_centroid, cur_rotated_c);
    }

    this->initer->add_vectors(rotated_centroids);

    delete rotated_centroids;
}

void IVF::allocate_memory(const std::vector<size_t>& cluster_sizes) {
    std::cout << "Allocating memory for IVF...\n";
    if (K < 20000ul) {
        this->initer = new FlatInitializer(D, K);
    } else {
        this->initer = new HNSWInitializer(D, K);
    }
    this->SHORT_DATA = memory::align_mm<64, uint8_t, true>(shortdata_bytes(cluster_sizes));
    this->LONG_CODE = memory::align_mm<64, uint8_t, true>(longcode_bytes());
    this->EX_FACTOR = memory::align_mm<64, ExFactor, true>(exfactor_bytes());
    this->IDs = memory::align_mm<64, PID, true>(ids_bytes());
}

void IVF::init_clusters(const std::vector<size_t>& cluster_sizes) {
    this->ClusterLst.reserve(K);
    size_t added_vectors = 0;
    size_t added_blocks = 0;
    for (size_t i = 0; i < K; ++i) {
        // find data location for current cluster
        size_t num = cluster_sizes[i];
        size_t num_blocks = DQ.num_blocks(num);

        uint8_t* short_data =
            SHORT_DATA + added_blocks * DQ.block_bytes() / sizeof(uint8_t);
        uint8_t* long_code = LONG_CODE + added_vectors * DQ.long_code_length();
        ExFactor* ex_fac = EX_FACTOR + added_vectors;
        PID* ids = IDs + added_vectors;

        Cluster cur_cluster(num, short_data, long_code, ex_fac, ids);
        this->ClusterLst.push_back(std::move(cur_cluster));

        added_vectors += num;
        added_blocks += num_blocks;
    }
}

void IVF::quantize_cluster(
    Cluster& cp,
    const std::vector<PID>& IDs,
    const float* data,
    const float* cur_centroid,
    float* rotated_c
) {
    size_t num = IDs.size();
    if (cp.num() != num) {
        std::cerr << "Size of cluster and IDs are inequivalent\n";
        std::cerr << "Cluster: " << cp.num() << " IDs: " << num << '\n';
    }
    /* Copy ids */
    PID* idp = cp.ids();
    std::copy(IDs.begin(), IDs.end(), idp);

    this->DQ.quantize(
        data,
        cur_centroid,
        IDs,
        this->Rota,
        cp.first_block(),
        cp.long_code(0, DQ),
        cp.ex_factor(0),
        rotated_c
    );
}

void IVF::save(const char* filename) const {
    if (ClusterLst.size() == 0) {
        std::cerr << "IVF not constructed\n";
        return;
    }

    std::ofstream output(filename, std::ios::binary);

    /* Save meta data */
    output.write((char*)&N, sizeof(size_t));
    output.write((char*)&DIM, sizeof(size_t));
    output.write((char*)&K, sizeof(size_t));
    output.write((char*)&EX_BITS, sizeof(size_t));

    /* Save number of vectors of each cluster */
    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(K);
    for (auto& cur_cluster : ClusterLst) {
        cluster_sizes.push_back(cur_cluster.num());
    }
    output.write((char*)cluster_sizes.data(), sizeof(size_t) * K);

    /* Save rotator */
    this->Rota.save(output);

    /* Save data */
    this->initer->save(output, filename);
    output.write((char*)SHORT_DATA, shortdata_bytes(cluster_sizes));
    output.write((char*)LONG_CODE, longcode_bytes());
    output.write((char*)EX_FACTOR, exfactor_bytes());
    output.write((char*)IDs, ids_bytes());

    output.close();
}

void IVF::load(const char* filename) {
    std::cout << "Loading IVF...\n";
    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());

    /* Load meta data */
    std::cout << "\tLoading meta data...\n";
    input.read((char*)&this->N, sizeof(size_t));
    input.read((char*)&this->DIM, sizeof(size_t));
    this->D = rd_up_to_multiple_of(this->DIM, 64);
    input.read((char*)&this->K, sizeof(size_t));
    input.read((char*)&this->EX_BITS, sizeof(size_t));

    /* Init rotator and quantizer */
    this->DQ = DataQuantizer(DIM, EX_BITS);
    this->Rota = Rotator(DIM);

    /* Load number of vectors of each cluster */
    std::vector<size_t> cluster_sizes(K, 0);
    input.read((char*)cluster_sizes.data(), sizeof(size_t) * K);
    assert(std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), size_t(0)) == N);

    /* Load rotator */
    this->Rota.load(input);

    /* Load data */
    free_memory();
    allocate_memory(cluster_sizes);
    this->initer->load(input, filename);
    input.read((char*)SHORT_DATA, shortdata_bytes(cluster_sizes));
    input.read((char*)LONG_CODE, longcode_bytes());
    input.read((char*)EX_FACTOR, exfactor_bytes());
    input.read((char*)IDs, ids_bytes());

    /* Init each cluster */
    init_clusters(cluster_sizes);

    input.close();
    std::cout << "Index loaded\n";
}

void IVF::search(
    const float* __restrict__ query,
    const float* __restrict__ data,
    size_t k,
    size_t nprobe,
    PID* __restrict__ results
) const {
    /* Compute distance to rotated centroids */
    std::vector<Candidate> centroid_dist(nprobe);
    this->initer->centroids_distances(query, nprobe, centroid_dist);

    ResultPool KNNs(k);
#if defined(HIGH_ACC_FAST_SCAN)
    HASearcher searcher(query, D, EX_BITS, DQ);
#else
    Searcher searcher(query, D, EX_BITS, DQ);
#endif

    for (size_t i = 0; i < nprobe; ++i) {
        // stopw.reset();
        PID cid = centroid_dist[i].id;
        float sqr_y = centroid_dist[i].distance;
        float* cur_centroid = this->initer->centroid(cid);
        const Cluster& cur_cluster = ClusterLst[cid];

        searcher.search_cluster(cur_cluster, cur_centroid, sqr_y, KNNs);
    }

    KNNs.copy_results(results);
}
