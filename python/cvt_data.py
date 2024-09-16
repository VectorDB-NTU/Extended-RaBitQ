# Convert data from fbin/ibin to fvecs/ivecs
import sys
from utils.io import fvecs_write, read_fbin

if __name__ == "__main__":
    dataset = sys.argv[1]

    base = read_fbin(f"./data/{dataset}/{dataset}_base.bin")
    query = read_fbin(f"./data/{dataset}/{dataset}_query.bin")

    fvecs_write(f"./data/{dataset}/{dataset}_base.fvecs", base)
    fvecs_write(f"./data/{dataset}/{dataset}_query.fvecs", query)