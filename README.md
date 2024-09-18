# Extended RaBitQ
Using Extended RaBitQ for approximate nearest neighbor search without storing raw data vectors in main memory

The project provides more trade-offs between space consumptions and the accuracy of distance estimation. Please refer to our paper for detailed information https://arxiv.org/pdf/2409.09913.

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
