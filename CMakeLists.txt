cmake_minimum_required (VERSION 3.10)

project(ExRaBitQ LANGUAGES CXX)

set(CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/inc)

# AVX512 required
include(CheckCXXSourceCompiles)
set(AVX2_CXX_CODE "#include <immintrin.h>
int main() {
#if __AVX512F__
return 0;
#else
#error \"AVX512F is not supported\"
#endif
}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS  "-Wall -Wextra -Ofast -lrt -std=c++17 -march=native -fpic -fopenmp -ftree-vectorize -fexceptions")

add_subdirectory(src)