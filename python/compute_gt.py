import numpy as np
import sys
from utils.io import read_fvecs, write_ivecs


if __name__ == "__main__":
    dataset = sys.argv[1]

    base = read_fvecs(f"./data/{dataset}/{dataset}_base.fvecs")
    query = read_fvecs(f"./data/{dataset}/{dataset}_query.fvecs")

    gt = []
    for q in query:
        distances = np.linalg.norm(base - q, axis=1)
        gt.append(list(np.argsort(distances))[:1000])

    gt = np.array(gt)

    write_ivecs(f"./data/{dataset}/{dataset}_groundtruth.ivecs", gt)
