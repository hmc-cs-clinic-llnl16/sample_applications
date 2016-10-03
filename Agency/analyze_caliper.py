#!/usr/bin python
from __future__ import division

import math
import os
import sys
import argparse
try:
    import subprocess32 as subprocess
except ImportError:
    import subprocess

import numpy as np
import scipy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument('cali_file', help="The cali file to be analyzed should go here")
args = parser.parse_args()

cali_file = args.cali_file

ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'initialization', 'control', 'n2Serial', 'n2Parallel', 'nSerial', 'nParallel', 'n10Serial', 'n10Parallel', 'time.inclusive.duration'])
CALI_QUERY = [os.path.join('build', 'tpl', 'bin', 'cali-query'),'-e', '--print-attributes={}'.format(ANNOTATIONS), cali_file]

p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
out, err = p.communicate()

data = {
    'init': {},
    'control': {},
    'n2Serial': {},
    'n2Parallel': {},
    'nSerial': {},
    'nParallel': {},
    'n10Serial': {},
    'n10Parallel': {}
}

for line in out.split():
    line = dict(map(lambda x: x.split('='), line.split(',')))
    loop = None
    time = None
    mode = None
    size = None
    for key, value in line.iteritems():
        if key == 'time.inclusive.duration':
            time = int(value)
        elif key == 'iteration':
            loop = int(value)
        elif key == 'size':
            size = int(value)
        else:
            mode = key

    if mode == 'initialization':
        data['init'][size] = time
    elif mode:
        if size not in data[mode]:
            data[mode][size] = [None] * 10

        if size is not None and loop is not None:
            data[mode][size][loop] = time

Control = data['control']
N2Serial = data['n2Serial']
N2Parallel = data['n2Parallel']
NSerial = data['nSerial']
NParallel = data['nParallel']
N10Serial = data['n10Serial']
N10Parallel = data['n10Parallel']

dat = [[size, control, n2serial, n2parallel, nserial, nparallel, n10serial, n10parallel]
       for size in N2Serial
       for control, n2serial, n2parallel, nserial, nparallel, n10serial, n10parallel in zip(Control[size],
                                                                                            N2Serial[size],
                                                                                            N2Parallel[size],
                                                                                            NSerial[size],
                                                                                            NParallel[size],
                                                                                            N10Serial[size],
                                                                                            N10Parallel[size])]
dataframe = pd.DataFrame(data=dat, columns=['Size', 'Control', 'n2Serial', 'n2Parallel', 'nSerial', 'nParallel', 'n10Serial', 'n10Parallel'])
d = [[size,
      dataframe[dataframe['Size'] == size]['Control'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Control']),
      dataframe[dataframe['Size'] == size]['n2Serial'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['n2Serial']),
      dataframe[dataframe['Size'] == size]['n2Parallel'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['n2Parallel']),
      dataframe[dataframe['Size'] == size]['nSerial'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['nSerial']),
      dataframe[dataframe['Size'] == size]['nParallel'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['nParallel']),
      dataframe[dataframe['Size'] == size]['n10Serial'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['n10Serial']),
      dataframe[dataframe['Size'] == size]['n10Parallel'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['n10Parallel'])
     ] for size in sorted(N2Serial)]
realDataframe = pd.DataFrame(data=d, columns=['Size',
                                              'Controlmean', 'Controlsem',
                                              'n2Serialmean', 'n2Serialsem',
                                              'n2Parallelmean', 'n2Parallelsem',
                                              'nSerialmean', 'nSerialsem',
                                              'nParallelmean', 'nParallelsem',
                                              'n10Serialmean', 'n10Serialsem',
                                              'n10Parallelmean', 'n10Parallelsem'])

sizes = sorted(N2Serial)

fig = plt.figure()
plt.errorbar(sizes, realDataframe['Controlmean'], color='g', xerr=[0]*len(sizes), yerr=realDataframe['Controlsem'], label='Control')
plt.errorbar(sizes, realDataframe['n2Serialmean'], color='#ff0000', xerr=[0]*len(sizes), yerr=realDataframe['n2Serialsem'], label='Serial CPU n^2 workers')
plt.errorbar(sizes, realDataframe['nSerialmean'], color='#C00000', xerr=[0]*len(sizes), yerr=realDataframe['nSerialsem'], label='Serial CPU n workers')
plt.errorbar(sizes, realDataframe['n10Serialmean'], color='#800000', xerr=[0]*len(sizes), yerr=realDataframe['n10Serialsem'], label='Serial CPU n/10 workers')
plt.errorbar(sizes, realDataframe['n2Parallelmean'], color='#0000ff', xerr=[0]*len(sizes), yerr=realDataframe['n2Parallelsem'], label='Parallel CPU n^2 workers')
plt.errorbar(sizes, realDataframe['nParallelmean'], color='#0000C0', xerr=[0]*len(sizes), yerr=realDataframe['nParallelsem'], label='Parallel CPU n workers')
plt.errorbar(sizes, realDataframe['n10Parallelmean'], color='#000080', xerr=[0]*len(sizes), yerr=realDataframe['n10Parallelsem'], label='Parallel CPU n/10 workers')
plt.legend(loc='lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('matrix size', fontsize=16)
plt.ylabel('time taken', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.title('Testing Agency Speed', fontsize=16)
plt.savefig('tmp.pdf')

