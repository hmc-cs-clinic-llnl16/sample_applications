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

ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'initialization', 'control', 'Serial', 'Parallel', 'time.inclusive.duration'])
CALI_QUERY = [os.path.join('build', 'tpl', 'bin', 'cali-query'),'-e', '--print-attributes={}'.format(ANNOTATIONS), cali_file]

p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
out, err = p.communicate()

data = {
    'init': {},
    'control': {},
    'Serial': {},
    'Parallel': {}
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
Serial = data['Serial']
Parallel = data['Parallel']

dat = [[size, control, serial, parallel]
       for size in Control
       for control, serial, parallel in zip(Control[size],
                                            Serial[size],
                                            Parallel[size])]
dataframe = pd.DataFrame(data=dat, columns=['Size', 'Control', 'Serial', 'Parallel'])
d = [[size,
      dataframe[dataframe['Size'] == size]['Control'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Control']),
      dataframe[dataframe['Size'] == size]['Serial'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Serial']),
      dataframe[dataframe['Size'] == size]['Parallel'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Parallel'])
     ] for size in sorted(Control)]
realDataframe = pd.DataFrame(data=d, columns=['Size',
                                              'Controlmean', 'Controlsem',
                                              'Serialmean', 'Serialsem',
                                              'Parallelmean', 'Parallelsem'])

sizes = sorted(Control)

fig = plt.figure()
plt.errorbar(sizes, realDataframe['Controlmean']/realDataframe['Controlmean'], color='g', xerr=[0]*len(sizes), yerr=realDataframe['Controlsem']/realDataframe['Controlmean'], label='Control')
plt.errorbar(sizes, realDataframe['Serialmean']/realDataframe['Controlmean'], color='r', xerr=[0]*len(sizes), yerr=realDataframe['Serialsem']/realDataframe['Controlmean'], label='Serial CPU 4 workers')
plt.errorbar(sizes, realDataframe['Parallelmean']/realDataframe['Controlmean'], color='b', xerr=[0]*len(sizes), yerr=realDataframe['Parallelsem']/realDataframe['Controlmean'], label='Parallel CPU 4 workers')
plt.legend(loc='lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('matrix size', fontsize=16)
plt.ylabel('time taken', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.title('Testing Agency Speed', fontsize=16)
plt.savefig('tmp.pdf')

