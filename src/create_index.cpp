#define HIGH_ACC_FAST_SCAN
#define EIGEN_DONT_PARALLELIZE
#include <fstream>
#include <iostream>

#include "index/IVF.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"

int main(int argc, char* argv[]) {
    assert(argc == 4);
    char* DATASET = argv[1];
    size_t K = atoi(argv[2]);
    int B = atoi(argv[3]);
    assert(B == 9 || B == 5 || B == 7 || B == 3 || B == 4 || B == 8);

    char data_file[500];
    char centroids_file[500];
    char cids_file[500];
    char ivf_file[500];
    char log_file[500];

    sprintf(data_file, "../data/%s/%s_base.fvecs", DATASET, DATASET);
    sprintf(centroids_file, "../data/%s/%s_centroid_%ld.fvecs", DATASET, DATASET, K);
    sprintf(cids_file, "../data/%s/%s_cluster_id_%ld.ivecs", DATASET, DATASET, K);
    sprintf(ivf_file, "../data/%s/ivf_exhaf%d.index", DATASET, B);
    sprintf(log_file, "../results/indexing_time/%s.csv", DATASET);

    FloatRowMat data;
    FloatRowMat centroids;
    UintRowMat cids;

    load_vecs<float, FloatRowMat>(data_file, data);
    load_vecs<float, FloatRowMat>(centroids_file, centroids);
    load_vecs<PID, UintRowMat>(cids_file, cids);

    size_t N = data.rows();
    size_t DIM = data.cols();

    std::cout << "data loaded\n";
    std::cout << "\tN: " << N << '\n';
    std::cout << "\tDIM: " << DIM << '\n';

    StopW stopw;
    IVF ivf(N, DIM, K, B);
    ivf.construct(data.data(), centroids.data(), cids.data());
    float miniutes = stopw.getElapsedTimeMili() / 1000 / 60;
    std::cout << "ivf constructed \n";
    ivf.save(ivf_file);

    std::cout << "Indexing time: " << miniutes << "miniutes\n";

    return 0;
}