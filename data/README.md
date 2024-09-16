# Datasets

##### Orgnization
Datasets and corresponding indices are stored in `./data/${dataset}`. For example, `./data/openai1536` contains `openai1536_base.fvecs`, `openai1536_query.fvecs`, `openai1536_groundtruth.ivecs` and indices.

##### Download and pre-processing
Use `python ./python/download_dataset.py` to download openai embeddings. Then run `python ./python/compute_gt.py openai1536` to generate groundtruth for KNN search.

More tested datasets are available at https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html.

Note: You need to convert the data format to `.fvecs` and `.ivecs` to run the tests smoothly. For datasets in `.fbin` and `.ibin` formats, we provide helper functions `./python/cvt_data.py` to help you transform the data. For datasets without groundtruth, please refer to `./python/compute_gt.py` to generate the groundtruth.

##### Clustering
Before building indices by Extended RaBitQ, you need to use python and faiss library to train an IVF index. Please edit the `DATASET` and number of clusters `K` in `./python/ivf.py` to build and save IVF for a certain dataset.
```
python ./python/ivf.py
```
Once the process is finished, corresponding data dir will contain centroid vectors of IVF clusters and the cluster id for each data vector. For example, `./data/openai1536/` will contain `openai1536_centroid_4096.fvecs` and `openai1536_cluster_id_4096.ivecs`, where `4096` is `K`. Then, you can use them to build the index.