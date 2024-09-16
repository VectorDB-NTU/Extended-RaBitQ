import faiss
import os
from utils.io import read_fvecs, fvecs_write, write_ivecs, read_fbin

SOURCE = "./data/"
DATASET = "openai1536"
K = 4096

if __name__ == "__main__":
    print(f"Clustering - {DATASET}")
    # path
    path = os.path.join(SOURCE, DATASET)

    data_path = os.path.join(path, f"{DATASET}_base.fvecs")
    X = read_fvecs(data_path)

    # data_path = os.path.join(path, f"{DATASET}_base.bin")
    # X = read_fbin(data_path)

    D = X.shape[1]
    centroids_path = os.path.join(path, f"{DATASET}_centroid_{K}.fvecs")
    dist_to_centroid_path = os.path.join(path, f"{DATASET}_dist_to_centroid_{K}.fvecs")
    cluster_id_path = os.path.join(path, f"{DATASET}_cluster_id_{K}.ivecs")

    # cluster data vectors
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    dist_to_centroid, cluster_id = index.quantizer.search(X, 1)
    dist_to_centroid = dist_to_centroid**0.5

    fvecs_write(dist_to_centroid_path, dist_to_centroid)
    write_ivecs(cluster_id_path, cluster_id)
    fvecs_write(centroids_path, centroids)
