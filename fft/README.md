#Fast Fourier Transform

This directory contains an implementation of the Fast Fourier Transform algorithm. 
It uses Bailey's Six Step Method for the parallel methods, and the Cooley-Tukey method for the serial versions.

There is a serial control, as well as a RAJA serial version. The parallel execution policies use RAJA's OpenMP, Agency, and Agency + OpenMP policies.

The data is collected using Caliper and then analyzed with the Python script `analyze_caliper.py`, resulting in a figure in the `figs` directory above.
