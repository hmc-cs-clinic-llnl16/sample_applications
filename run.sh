#!/bin/bash

if [ ! -d build ]; then
  mkdir build
fi
 
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=tpl -DPYTHON_EXECUTABLE=$(which python2) \
         -DRAJA_ENABLE_CUDA=$RAJA_ENABLE_CUDA -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR
make clean
make

#cd matrix_multiplication
#CALI_CONFIG_PROFILE=thread-trace CALI_LOG_VERBOSITY=2 ./mmult.exe
#../tpl/bin/cali-query -e \
#    --print-attributes=iteration:size:loop:initialization:control:test:Serial:OMP:time.inclusive.duration \
#   $(find ./*.cali | tail -n 1) 

