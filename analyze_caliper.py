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
mpl.use('pdf')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--cali_file', help="The cali file to be analyzed should go here")
    parser.add_argument('parallel_type', choices=['raja', 'agency'], help="Parallelism framework being used.")
    parser.add_argument('application_type', choices=['fft', 'mmult'], help="Application benchmark being plotted")
    parser.add_argument('-o', '--output_file', help="Name of the output file")
    parser.add_argument('-q', '--cali_query_loc', help="Path to cali query executable")

    args = parser.parse_args()
    if args.parallel_type == 'raja' and args.application_type == 'mmult':
        print args.cali_query_loc
        plot_raja_mmult(args.cali_file, args.output_file, args.cali_query_loc)
    elif args.parallel_type == 'agency' and args.application_type == 'fft':
        plot_agency_fft(args.cali_file, args.output_file, args.cali_query_loc)
    else:
        sys.exit("Parallel framework {} and application {} not yet supported.".format(args.parallel_type, args.application_type))

def plot_raja_mmult(cali_file, filename, cali_query):
    ANNOTATIONS = ':'.join(['iteration', 'size', 'initialization', 'control', 'Sequential', 'Parallel', 'time.inclusive.duration'])
    CALI_QUERY = [cali_query, '-e', '--print-attributes={}'.format(ANNOTATIONS), cali_file]

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
        elif mode:
            if size not in data[mode]:
                data[mode][size] = [None] * 10

            if size is not None and loop is not None:
                data[mode][size][loop] = time

    OMP = data['OMP']
    control = data['control']
    Serial = data['Serial']

    dat = [[size, c, omp, serial] for size in OMP for omp, serial, c in zip(OMP[size], Serial[size], control[size])]
    dataframe = pd.DataFrame(data=dat, columns=['Size', 'Control', 'OMP', 'Serial'])
    d = [[size, 
          dataframe[dataframe['Size'] == size]['OMP'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['OMP']),
          dataframe[dataframe['Size'] == size]['Serial'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['Serial']),
          dataframe[dataframe['Size'] == size]['Control'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['Control'])
         ] for size in sorted(OMP)]
    realDataframe = pd.DataFrame(data=d, columns=['Size', 'OMPmean', 'OMPsem', 'Serialmean', 'Serialsem', 'controlmean', 'controlsem'])

    fig = plt.figure()
    sizes = sorted(OMP)
    legendNames = []
    plt.errorbar(sizes, realDataframe['OMPmean'], color='r', xerr=[0]*len(sizes), yerr=realDataframe['OMPsem']) 
    legendNames.append('OMP RAJA')
    plt.errorbar(sizes, realDataframe['Serialmean'], color='b', xerr=[0]*len(sizes), yerr=realDataframe['Serialsem']) 
    legendNames.append('Serial RAJA')
    plt.errorbar(sizes, realDataframe['controlmean'], color='g', xerr=[0]*len(sizes), yerr=realDataframe['controlsem'])
    legendNames.append('control')
    plt.legend(legendNames, loc='lower right')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('matrix size', fontsize=16)
    plt.ylabel('time taken', fontsize=16)
    plt.grid(b=True, which='major', color='k', linestyle='dotted')
    plt.title('Testing RAJA speed', fontsize=16)
    plt.savefig(filename)



def plot_agency_fft(cali_file, filename, cali_query):
    ANNOTATIONS = ':'.join(['iteration', 'size', 'initialization', 'control', 'Sequential', 'Parallel', 'time.inclusive.duration'])
    CALI_QUERY = [cali_query, '-e', '--print-attributes={}'.format(ANNOTATIONS), cali_file]

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
        elif mode:
            if size not in data[mode]:
                data[mode][size] = [None] * 10

            if size is not None and loop is not None:
                data[mode][size][loop] = time

    parallel = data['Parallel']
    control = data['control']
    Serial = data['Sequential']

    dat = [[size, c, omp, serial] for size in parallel for omp, serial, c in zip(parallel[size], Serial[size], control[size])]
    dataframe = pd.DataFrame(data=dat, columns=['Size', 'Control', 'parallel', 'Serial'])
    d = [[size, 
          dataframe[dataframe['Size'] == size]['parallel'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['parallel']),
          dataframe[dataframe['Size'] == size]['Serial'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['Serial']),
          dataframe[dataframe['Size'] == size]['Control'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['Control'])
         ] for size in sorted(parallel)]
    realDataframe = pd.DataFrame(data=d, columns=['Size', 'OMPmean', 'OMPsem', 'Serialmean', 'Serialsem', 'controlmean', 'controlsem'])

    fig = plt.figure()
    sizes = sorted(parallel)
    legendNames = []
    plt.errorbar(sizes, realDataframe['OMPmean'], color='r', xerr=[0]*len(sizes), yerr=realDataframe['OMPsem']) 
    legendNames.append('Parallel Agency')
    plt.errorbar(sizes, realDataframe['Serialmean'], color='b', xerr=[0]*len(sizes), yerr=realDataframe['Serialsem']) 
    legendNames.append('Serial Agency')
    plt.errorbar(sizes, realDataframe['controlmean'], color='g', xerr=[0]*len(sizes), yerr=realDataframe['controlsem'])
    legendNames.append('control')
    plt.legend(legendNames, loc='lower right')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Vector size', fontsize=16)
    plt.ylabel('time taken', fontsize=16)
    plt.grid(b=True, which='major', color='k', linestyle='dotted')
    plt.title('Testing Agency speed', fontsize=16)
    plt.savefig(filename)


if __name__ == '__main__':
    main()
