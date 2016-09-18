#!/bin/bash

if [ ! -d build ]; then
  mkdir build
fi
 
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=tpl -DPYTHON_EXECUTABLE=$(which python2) \
         -DRAJA_ENABLE_CUDA=$RAJA_ENABLE_CUDA -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR
make clean
make

cd matrix_multiplication
CALI_CONFIG_PROFILE=thread-trace CALI_LOG_VERBOSITY=2 ./mmult.exe
cd ../..
python ./analyze_caliper.py $(find ./build/matrix_multiplication/*.cali | tail -n 1)
