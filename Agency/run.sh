#!/bin/bash

if [ ! -d ../build ]; then
  mkdir ../build
fi

cd ../build
cmake .. -DCMAKE_INSTALL_PREFIX=tpl -DPYTHON_EXECUTABLE=$(which python2) \
         -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR
make clean
make mmultAgency.exe

cd Agency
CALI_CONFIG_PROFILE=thread-trace CALI_LOG_VERBOSITY=2 ./mmultAgency.exe
cd ../..
../anaconda2/bin/python2.7 Agency/analyze_caliper.py $(find ./build/Agency/*.cali | tail -n 1)
