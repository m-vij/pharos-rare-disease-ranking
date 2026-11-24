#!/bin/bash
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -p defq
#SBATCH --time=01:30:00
#SBATCH -o build_llama3.out
module unload gcc/11.2.0 2>/dev/null
module load gcc/15.2.0
module load cmake/3.23.2
module load cuda12.3/toolkit/12.3.2
GCC11=/cm/local/apps/gcc/11.2.0/bin
echo "--- toolchain ---"
which gcc g++ cmake nvcc
g++ --version | head -n 1
$GCC11/g++ --version | head -n 1
nvcc --version | tail -n 2
echo "--- commit ---"
cd ~/PHAROS/llama.cpp
git rev-parse HEAD
echo "--- build ---"
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_CUDA_HOST_COMPILER=$GCC11/g++
cmake --build build --config Release -j 8 --target llama-server llama-cli
ls -la build/bin/llama-server build/bin/llama-cli
