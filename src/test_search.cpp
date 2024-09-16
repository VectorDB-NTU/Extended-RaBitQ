#define HIGH_ACC_FAST_SCAN
#define EIGEN_DONT_PARALLELIZE
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "defines.hpp"
#include "index/IVF.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"

size_t TOPK = 100;
size_t ROUND = 3;

std::vector<size_t> get_nprobes(
    const IVF& ivf,
    const std::vector<size_t>& all_nprobes,
    FloatRowMat& rotated_query,
    FloatRowMat& data,
    UintRowMat& gt
);
int main(int argc, char* argv[]) {
    assert(argc == 3);
    char* DATASET = argv[1];
    int B = atoi(argv[2]);
    assert(B == 9 || B == 5 || B == 7 || B == 3 || B == 4 || B == 8);

    char data_file[500];
    char query_file[500];
    char gt_file[500];
    char ivf_file[500];
    char result_file[500];

    sprintf(data_file, "../data/%s/%s_base.fvecs", DATASET, DATASET);
    sprintf(query_file, "../data/%s/%s_query.fvecs", DATASET, DATASET);
    sprintf(gt_file, "../data/%s/%s_groundtruth.ivecs", DATASET, DATASET);
    sprintf(ivf_file, "../data/%s/ivf_exhaf%d.index", DATASET, B);
    sprintf(result_file, "../results/exrabitq/%s_exhaf%d.csv", DATASET, B);

    FloatRowMat data;
    FloatRowMat query;
    UintRowMat gt;

    load_vecs<float, FloatRowMat>(data_file, data);
    load_vecs<float, FloatRowMat>(query_file, query);
    load_vecs<PID, UintRowMat>(gt_file, gt);

    size_t N = data.rows();
    size_t DIM = data.cols();
    size_t NQ = query.rows();

    std::cout << "data loaded\n";
    std::cout << "\tN: " << N << '\n' << "\tDIM: " << DIM << '\n';
    std::cout << "query loaded\n";
    std::cout << "\tNQ: " << NQ << '\n';

    IVF ivf;
    ivf.load(ivf_file);

    std::vector<size_t> all_nprobes;
    all_nprobes.push_back(5);
    for (size_t i = 10; i < 200; i += 10) {
        all_nprobes.push_back(i);
    }
    for (size_t i = 200; i < 400; i += 40) {
        all_nprobes.push_back(i);
    }
    for (size_t i = 400; i <= 1500; i += 100) {
        all_nprobes.push_back(i);
    }
    for (size_t i = 2000; i <= 4000; i += 500) {
        all_nprobes.push_back(i);
    }

    all_nprobes.push_back(6000);
    all_nprobes.push_back(10000);
    all_nprobes.push_back(15000);

    size_t total_count = NQ * TOPK;
    StopW stopw;

    FloatRowMat padded_query(NQ, ivf.padded_dim());
    padded_query.setZero();
    FloatRowMat rotated_query(NQ, ivf.padded_dim());
    for (size_t i = 0; i < NQ; ++i) {
        std::memcpy(&padded_query(i, 0), &query(i, 0), sizeof(float) * DIM);
    }
    Rotator& rp = ivf.rotator();
    stopw.reset();
    rp.rotate(padded_query, rotated_query);
    float rotate_time = stopw.getElapsedTimeMicro();

    auto nprobes = get_nprobes(ivf, all_nprobes, rotated_query, data, gt);
    size_t length = nprobes.size();

    std::vector<std::vector<float>> all_qps(ROUND, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(ROUND, std::vector<float>(length));
    std::vector<std::vector<float>> all_ratio(ROUND, std::vector<float>(length));

    for (size_t r = 0; r < ROUND; r++) {
        for (size_t i = 0; i < length; ++i) {
            size_t nprobe = nprobes[i];
            size_t total_correct = 0;
            double total_ratio = 0;
            float total_time = 0;
            PID results[TOPK];
            for (size_t i = 0; i < NQ; i++) {
                stopw.reset();
                ivf.search(&rotated_query(i, 0), data.data(), TOPK, nprobe, results);
                total_time += stopw.getElapsedTimeMicro();
                total_ratio += get_ratio(i, query, data, gt, results, TOPK, L2Sqr);
                for (size_t j = 0; j < TOPK; j++) {
                    for (size_t k = 0; k < TOPK; k++) {
                        if (gt(i, k) == results[j]) {
                            total_correct++;
                            break;
                        }
                    }
                }
            }
            float qps = NQ / ((total_time + rotate_time) / 1e6);
            float recall = static_cast<float>(total_correct) / total_count;
            float ratio = total_ratio / total_count;

            all_qps[r][i] = qps;
            all_recall[r][i] = recall;
            all_ratio[r][i] = ratio;
        }
    }

    auto avg_qps = horizontal_avg(all_qps);
    auto avg_recall = horizontal_avg(all_recall);
    auto avg_ratio = horizontal_avg(all_ratio);

    std::ofstream csv_data(result_file, std::ios::out);
    csv_data << "nprobe,QPS,recall,ratio" << std::endl;

    for (size_t i = 0; i < length; ++i) {
        size_t nprobe = nprobes[i];
        float qps = avg_qps[i];
        float recall = avg_recall[i];
        float ratio = avg_ratio[i];

        csv_data << nprobe << ',';
        csv_data << qps << ',';
        csv_data << recall << ',';
        csv_data << ratio << '\n';
    }
    csv_data.close();

    return 0;
}

std::vector<size_t> get_nprobes(
    const IVF& ivf,
    const std::vector<size_t>& all_nprobes,
    FloatRowMat& rotated_query,
    FloatRowMat& data,
    UintRowMat& gt
) {
    StopW stopw;
    size_t NQ = rotated_query.rows();
    size_t total_count = TOPK * NQ;
    float old_recall = 0;
    std::vector<size_t> nprobes;

    for (auto nprobe : all_nprobes) {
        if (nprobe > ivf.k()) {
            break;
        }
        nprobes.push_back(nprobe);

        size_t total_correct = 0;
        float total_time = 0;
        PID results[TOPK];
        for (size_t i = 0; i < NQ; i++) {
            stopw.reset();
            ivf.search(&rotated_query(i, 0), data.data(), TOPK, nprobe, results);
            total_time += stopw.getElapsedTimeMicro();
            for (size_t j = 0; j < TOPK; j++) {
                for (size_t k = 0; k < TOPK; k++) {
                    if (gt(i, k) == results[j]) {
                        total_correct++;
                        break;
                    }
                }
            }
        }
        float recall = static_cast<float>(total_correct) / total_count;
        if (recall > 0.997 || recall - old_recall < 1e-5) {
            break;
        }
        std::cout << recall << '\t' << nprobe << std::endl << std::flush;
        old_recall = recall;
    }

    return nprobes;
}