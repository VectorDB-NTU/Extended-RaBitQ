# Extended RaBitQ

News: The paper has been accepted by SIGMOD 2025.

[SIGMOD 2025] Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
> Replace your scalar and binary quantization with RaBitQ seamlessly. Enjoy blazingly fast distance computation with dominant accuracy. 

The project proposes a novel quantization algorithm developped from RaBitQ. The algorithm supports to compress high-dimensional vectors with arbitrary compression rates. Its computation is exactly the same as the classical scalar quantization and has dominant accuracy under same compression rates. It brings especially significant improvement in the setting from 2-bit to 6-bit, which helps an algorithm to achieve high recall without reranking.
We summarize the key intuitions and results as follows. For more details, please refer to our paper https://arxiv.org/pdf/2409.09913.

*   Intuitively, the algorithm achieves promising performance because it "constructs" the codebook for every vector individually. Specifically, our method first considers a codebook formed by $B$-bit uint vectors; and then transform the codebook, i.e., translating and rescaling, on a per-vector basis. By trying different parameters of transforming, the algorithm can find a vector in the transformed codebook, that best approximate a data vector.
*   Geometrically, the operation is equivalent to (1) normalizing both data vectors and the vectors in codebooks onto high-dim unit spheres and; (2)find every data vector's best match in the codebook after the normalization operation.
*   Theoretically, we prove that based on the technology above, our algorithm provides an asymptotically optimal error bound regarding the trade-off between the error and the space consumption. To our knowledge, our algorithm is also the first practical algorithm that achieves the optimality. 
*   The computation of distances and inner product can be fully supported by the arithmetic operations of unsigned integers, i.e., the computation is exactly the same as the classical scalar quantization. Thus, it can be seen as a better scalar quantization and replace the classical ones seamlessly.

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
