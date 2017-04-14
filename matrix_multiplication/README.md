#Matrix Multiplication

This directory contains an implementation of the naive matrix multiplication algorithm, configured in a variety of different ways.
Compiling `mmult.cxx` will create an executable that tests these configurations:

- Control Serial
- Control Parallel (OpenMP)
- RAJA Serial
- RAJA OpenMP
- RAJA Agency
- RAJA Agency OpenMP

All of the RAJA versions are tested using a `forallN` of depth 3 and of depth 2, as well as using a bare `forall`.
Furthermore, all of the above configurations are run using both an IJK and an IKJ loop ordering.

The data is collected using Caliper and then analyzed with the Python script `analyze_caliper.py`, resulting in a number of figures
in the `figs` directory above.
