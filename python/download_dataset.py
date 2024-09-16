from datasets import load_dataset
import os
import numpy as np
import struct

ds = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M")
name = "openai1536"

print("loaded dataset!")

data_dir = f"./data/{name}/"
try:
    os.makedirs(data_dir)
except OSError as e:
    print(e)

n = ds["train"].shape[0]
d = len(ds["train"][0]["text-embedding-3-large-1536-embedding"])
sequence = np.random.permutation(n)

query_ids = set(sequence[:1000])

with open(data_dir + f"{name}_query.fvecs", "wb") as f1, open(
    data_dir + f"{name}_base.fvecs", "wb"
) as f2:
    i = 0
    for row in ds["train"]:
        embd = row["text-embedding-3-large-1536-embedding"]
        if i in query_ids:
            f1.write(struct.pack("i", d))
            f1.write(struct.pack(f"{d}f", *embd))
        else:
            f2.write(struct.pack("i", d))
            f2.write(struct.pack(f"{d}f", *embd))
        i += 1
