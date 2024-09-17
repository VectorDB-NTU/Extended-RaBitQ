# Convert data from fbin/ibin to fvecs/ivecs
import sys
from utils.io import write_fvecs, read_fbin

if __name__ == "__main__":
    dataset = sys.argv[1]

    base = read_fbin(f"./data/{dataset}/{dataset}_base.bin")
    query = read_fbin(f"./data/{dataset}/{dataset}_query.bin")

    write_fvecs(f"./data/{dataset}/{dataset}_base.fvecs", base)
    write_fvecs(f"./data/{dataset}/{dataset}_query.fvecs", query)