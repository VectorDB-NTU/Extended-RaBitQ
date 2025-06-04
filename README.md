# Extended RaBitQ

News: A library with more practical implementation techniques about RaBitQ is released at the [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library).

News: The paper (arXiv:2409.09913, September, 2024) has been accepted by SIGMOD 2025.


[SIGMOD 2025] Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
> Replace your scalar and binary quantization with RaBitQ seamlessly. Enjoy blazingly fast distance computation with dominant accuracy. 

The project proposes a novel quantization algorithm developped from RaBitQ. The algorithm supports to compress high-dimensional vectors with arbitrary compression rates. Its computation is exactly the same as the classical scalar quantization and has dominant accuracy under same compression rates. It brings especially significant improvement in the setting from 2-bit to 6-bit, which helps an algorithm to achieve high recall without reranking.
We summarize the key intuitions and results as follows. For more details, please refer to our paper https://arxiv.org/pdf/2409.09913.

### Prepapring 


##### Prerequisites
* Please refer to `./inc/third/README.md` for detailed information about third-party libraries.
* AVX512 is required

##### Compiling
```Bash
mkdir build bin
cd ./build
cmake ..
make
```
Source codes are stored in `./src`, binary files are stored in `./bin` please update the cmake file in `./src` after adding new source files.

#### Datasets
Download and preprocess the datasets. Detailed instructions can be found in `./data/README.md`.


### Creating index
```Base
cd bin/
./create_index openai1536 4096 4
./create_index openai1536 4096 8
```
* `openai1536` for the name of dataset
* `4096` for the number of clusters in IVF
* `4` and `8` for total number of bits used in ExRaBitQ per dimension. Current, we support `3,4,5,7,8,9` bits to quantize each dimension for different precision requirements.

### Test query performance
```Base
cd bin/
./test_search openai1536 4
./test_search openai1536 8
```
* The result files are stored in `./results/exrabitq/`
* Note: currently in the test code, we compute the average distance ratio so the raw datasets are loaded in memory.
