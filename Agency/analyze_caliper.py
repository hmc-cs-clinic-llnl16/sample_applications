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
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument('cali_file', help="The cali file to be analyzed should go here")
args = parser.parse_args()

cali_file = args.cali_file

ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'initialization', 'control', 'Sequential', 'Parallel', 'time.inclusive.duration'])
CALI_QUERY = [os.path.join('build', 'tpl', 'bin', 'cali-query'),'-e', '--print-attributes={}'.format(ANNOTATIONS), cali_file]

p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
out, err = p.communicate()

data = {
    'init': {},
    'control': {},
    'Parallel': {},
    'Sequential': {}
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

parallels = data['Parallel']
controls = data['control']
sequentials = data['Sequential']

dat = [[size, controls[size], parallel, sequential] for size in parallels for parallel, sequential in zip(parallels[size], sequentials[size])]
dataframe = pd.DataFrame(data=dat, columns=['Size', 'Control', 'Parallel', 'Sequential'])
d = [[size,
      control[size],
      dataframe[dataframe['Size'] == size]['Parallel'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Parallel']),
      dataframe[dataframe['Size'] == size]['Sequential'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Sequential'])
     ] for size in sorted(OMP)]
realDataframe = pd.DataFrame(data=d, columns=['Size', 'Control', 'Parallelmean', 'Parallelsem', 'Sequentialmean', 'Sequentialsem'])

fig = plt.figure()
sizes = sorted(Parallels)
legendNames = []
plt.errorbar(sizes, realDataframe['Parallelmean'], color='r', xerr=[0]*len(sizes), yerr=realDataframe['Parallelsem'])
legendNames.append('Parallel CPU Agency')
plt.errorbar(sizes, realDataframe['Sequentialmean'], color='b', xerr=[0]*len(sizes), yerr=realDataframe['Sequentialsem'])
legendNames.append('Sequential Agency')
plt.plot(sizes, control.values(), color='g')
legendNames.append('Sequential Control')
plt.legend(legendNames, loc='lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('matrix size', fontsize=16)
plt.ylabel('time taken', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
plt.title('Testing Agency speed', fontsize=16)
plt.savefig('tmp.pdf')

