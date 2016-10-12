#!/bin/bash

if [ ! -d build ]; then
      mkdir build
fi
   
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=tpl -DPYTHON_EXECUTABLE=$(which python2) \
         -DRAJA_ENABLE_CUDA=$RAJA_ENABLE_CUDA -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR
cd fft_agency
rm *.cali
make clean
make

CALI_CONFIG_PROFILE=thread-trace CALI_LOG_VERBOSITY=2 ./fft

python ../../analyze_caliper.py -f $(find ./*.cali | tail -n 1) -o fftAgencyFig -q ../tpl/bin/cali-query agency fft



