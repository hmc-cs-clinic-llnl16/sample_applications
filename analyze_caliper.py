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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument('cali_file', help="The cali file to be analyzed should go here")
args = parser.parse_args()

cali_file = args.cali_file

ANNOTATIONS = ':'.join(['iteration', 'control', 'size', 'Serial', 'OpenMP', 'Agency', 'AgencyOmp', 'time.inclusive.duration'])
CALI_QUERY = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'tpl', 'bin', 'cali-query'),'-e', '--print-attributes={}'.format(ANNOTATIONS), cali_file]

print CALI_QUERY
p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
out, err = p.communicate()

data = {
    'control': {},
    'OpenMP': {},
    'Serial': {},
    'Agency': {},
    'AgencyOmp': {}
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
        elif key in data:
            mode = key

    if mode in data:
      if size not in data[mode]:
          data[mode][size] = [time]
      else:
          data[mode][size].append(time)

OMP = data['OpenMP']
control = data['control']
Serial = data['Serial']
Agency = data['Agency']
AgencyOmp = data['AgencyOmp']

dat = [[size, c, omp, serial, agency, aomp] 
       for size in OMP for omp, c, serial, agency, aomp
       in zip(OMP[size], control[size], Serial[size], Agency[size], AgencyOmp[size])]
dataframe = pd.DataFrame(data=dat, columns=['Size', 'Control', 'OMP', 'Serial', 'Agency', 'AgencyOmp'])
d = [[size, 
      dataframe[dataframe['Size'] == size]['Control'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Control']),
      dataframe[dataframe['Size'] == size]['OMP'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['OMP']),
      dataframe[dataframe['Size'] == size]['Serial'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Serial']),
      dataframe[dataframe['Size'] == size]['Agency'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['Agency']),
      dataframe[dataframe['Size'] == size]['AgencyOmp'].mean(),
      stats.sem(dataframe[dataframe['Size'] == size]['AgencyOmp'])
     ] for size in sorted(OMP)]
realDataframe = pd.DataFrame(data=d, columns=['Size', 'Controlmean', 'Controlsem', 'OMPmean', 'OMPsem', 'Serialmean', 'Serialsem', 'Agencymean', 'Agencysem', 'Aompmean', 'Aompsem'])

fig = plt.figure()
sizes = sorted(OMP)
methods = (('Control', 'Controlmean', 'Controlsem'),
           ('RAJA w/ OpenMP', 'OMPmean', 'OMPsem'),
           ('RAJA Serial', 'Serialmean', 'Serialsem'),
           ('RAJA w/ Agency', 'Agencymean', 'Agencysem'),
           ('RAJA w/ Agency w/ OpenMP', 'Aompmean', 'Aompsem'))
colors = cm.jet(np.linspace(0, 1, len(methods)))
markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
xerr=[0]*len(sizes)
legendNames = []

for i, (legendName, meanKey, semKey) in enumerate(methods):
    plt.errorbar(sizes, realDataframe[meanKey], color=colors[i], xerr=xerr, yerr=realDataframe[semKey], marker=markers[i])
    legendNames.append(legendName)

plt.legend(legendNames, loc='best', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Signal length', fontsize=16)
plt.ylabel('Time taken (ms)', fontsize=16)
plt.grid(b=True, which='major', color='k', linestyle='dotted')
# plt.title('Testing RAJA speed', fontsize=16)
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs', 'rajaFft1D.pdf'))
