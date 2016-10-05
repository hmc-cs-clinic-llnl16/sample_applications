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

ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'initialization', 'control', 'Serial1Worker', 'Parallel1Worker', 'Serial2Worker', 'Parallel2Worker', 'Serial4Worker', 'Parallel4Worker', 'time.inclusive.duration'])
CALI_QUERY = [os.path.join('build', 'tpl', 'bin', 'cali-query'),'-e', '--print-attributes={}'.format(ANNOTATIONS), cali_file]

p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
out, err = p.communicate()

data = {
    'init': {},
    'control': {},
    'Serial1Worker': {},
    'Parallel1Worker': {},
    'Serial2Worker': {},
    'Parallel2Worker': {},
    'Serial4Worker': {},
    'Parallel4Worker': {}
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
        elif key == 'control' and mode:
            continue
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
OSerial = data['Serial1Worker']
OParallel = data['Parallel1Worker']
TSerial = data['Serial2Worker']
TParallel = data['Parallel2Worker']
FSerial = data['Serial4Worker']
FParallel = data['Parallel4Worker']

dat = [[size, control, oserial, oparallel, tserial, tparallel, fserial, fparallel]
       for size in Control
       for control, oserial, oparallel, tserial, tparallel, fserial, fparallel in zip(Control[size],
                                                                                      OSerial[size],
                                                                                      OParallel[size],
                                                                                      TSerial[size],
                                                                                      TParallel[size],
                                                                                      FSerial[size],
                                                                                      FParallel[size])]
dataframe = pd.DataFrame(data=dat, columns=['Size', 'Control', '1Serial', '1Parallel', '2Serial', '2Parallel', '4Serial', '4Parallel'])
d = [[size,
      dataframe[dataframe['Size'] == size]['Control'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Control']),
      dataframe[dataframe['Size'] == size]['1Serial'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['1Serial']),
      dataframe[dataframe['Size'] == size]['1Parallel'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['1Parallel']),
      dataframe[dataframe['Size'] == size]['2Serial'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['2Serial']),
      dataframe[dataframe['Size'] == size]['2Parallel'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['2Parallel']),
      dataframe[dataframe['Size'] == size]['4Serial'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['4Serial']),
      dataframe[dataframe['Size'] == size]['4Parallel'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['4Parallel'])
     ] for size in sorted(Control)]
realDataframe = pd.DataFrame(data=d, columns=['Size',
                                              'Controlmean', 'Controlsem',
                                              '1Serialmean', '1Serialsem',
                                              '1Parallelmean', '1Parallelsem',
                                              '2Serialmean', '2Serialsem',
                                              '2Parallelmean', '2Parallelsem',
                                              '4Serialmean', '4Serialsem',
                                              '4Parallelmean', '4Parallelsem'])

sizes = sorted(Control)

fig = plt.figure()
plt.errorbar(sizes, realDataframe['Controlmean'], color='g', xerr=[0]*len(sizes), yerr=realDataframe['Controlsem'], label='Control')
plt.errorbar(sizes, realDataframe['1Serialmean'], color='#ff0000', xerr=[0]*len(sizes), yerr=realDataframe['1Serialsem'], label='Serial CPU 1 worker')
plt.errorbar(sizes, realDataframe['2Serialmean'], color='#C00000', xerr=[0]*len(sizes), yerr=realDataframe['2Serialsem'], label='Serial CPU 4 workers')
plt.errorbar(sizes, realDataframe['4Serialmean'], color='#800000', xerr=[0]*len(sizes), yerr=realDataframe['4Serialsem'], label='Serial CPU 16 workers')
plt.errorbar(sizes, realDataframe['1Parallelmean'], color='#0000ff', xerr=[0]*len(sizes), yerr=realDataframe['1Parallelsem'], label='Parallel CPU 1 worker')
plt.errorbar(sizes, realDataframe['2Parallelmean'], color='#0000C0', xerr=[0]*len(sizes), yerr=realDataframe['2Parallelsem'], label='Parallel CPU 4 workers')
plt.errorbar(sizes, realDataframe['4Parallelmean'], color='#000080', xerr=[0]*len(sizes), yerr=realDataframe['4Parallelsem'], label='Parallel CPU 16 workers')
plt.legend(loc='lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('matrix size', fontsize=16)
plt.ylabel('time taken', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.title('Testing Agency Speed', fontsize=16)
plt.savefig('tmp.pdf')

