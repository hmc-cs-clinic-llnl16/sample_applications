#!/usr/bin python
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
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

cali_file = os.path.join('build', 'matrix_multiplication', '160915-121651_10997_SwvkRFzqy9mC.cali')

ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'initialization', 'control', 'Serial', 'OMP', 'time.inclusive.duration'])
CALI_QUERY = [os.path.join('build', 'tpl', 'bin', 'cali-query'),'-e', '--print-attributes={}'.format(ANNOTATIONS), cali_file]

p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
out, err = p.communicate()

data = {
    'init': {},
    'control': {},
    'OMP': {},
    'Serial': {}
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
    elif mode == 'control':
        data['control'][size] = time
    elif mode:
        if size not in data[mode]:
            data[mode][size] = [None] * 10

        if size is not None and loop is not None:
            data[mode][size][loop] = time

OMP = data['OMP']
control = data['control']
Serial = data['Serial']

dat = [[size, control[size], omp, serial] for size in OMP for omp, serial in zip(OMP[size], Serial[size])]
dataframe = pd.DataFrame(data=dat, columns=['Size', 'Control', 'OMP', 'Serial'])
d = [[size, 
      control[size], 
      dataframe[dataframe['Size'] == size]['OMP'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['OMP']),
      dataframe[dataframe['Size'] == size]['Serial'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Serial'])
     ] for size in sorted(OMP)]
realDataframe = pd.DataFrame(data=d, columns=['Size', 'Control', 'OMPmean', 'OMPsem', 'Serialmean', 'Serialsem'])

fig = plt.figure()
sizes = sorted(OMP)
plt.errorbar(sizes, realDataframe['OMPmean'], color='r', xerr=[0]*len(sizes), yerr=realDataframe['OMPsem']) 
plt.errorbar(sizes, realDataframe['Serialmean'], color='b', xerr=[0]*len(sizes), yerr=realDataframe['Serialsem']) 
plt.plot(sizes, control.values(), color='g')
plt.plot()
plt.savefig('tmp.pdf')
