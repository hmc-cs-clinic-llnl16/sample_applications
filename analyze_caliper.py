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
import pandas as pd
from scipy import stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--cali_file', help="The cali file to be analyzed should go here")
    parser.add_argument('parallel_type', choices=['raja', 'agency'], help="Parallelism framework being used.")
    parser.add_argument('application_type', choices=['fft', 'mmult', 'fft2d', 'reducer'], help="Application benchmark being plotted")
    parser.add_argument('-o', '--output_file', help="Name of the output file")
    parser.add_argument('-q', '--cali_query_loc', help="Path to cali query executable")

    args = parser.parse_args()
    if args.parallel_type == 'raja' and args.application_type == 'mmult':
        plot_raja_mmult(args.cali_file, args.output_file, args.cali_query_loc)
    elif args.parallel_type == 'raja' and args.application_type == 'fft2d':
        plot_raja_mmult(args.cali_file, args.output_file, args.cali_query_loc)
    elif args.parallel_type == 'raja' and args.application_type == 'reducer':
        plot_raja_reducer(args.cali_file, args.output_file, args.cali_query_loc)
    else:
        sys.exit("Parallel framework {} and application {} not yet supported.".format(args.parallel_type, args.application_type))

def _get_cali_query(cali_query, annotations, cali_file):
    return [cali_query, '-e', '--print-attributes={}'.format(annotations), cali_file]


def plot_raja_mmult(cali_file, filename, cali_query):
    ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'initialization', 'control', 'Serial', 'OMP', 'time.inclusive.duration'])
    CALI_QUERY = _get_cali_query(cali_query, ANNOTATIONS, cali_file)

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
    plt.errorbar(sizes, realDataframe['OMPmean']/realDataframe['controlmean'], color='r', xerr=[0]*len(sizes), yerr=realDataframe['OMPsem']/realDataframe['controlmean'])
    legendNames.append('OMP RAJA')
    plt.errorbar(sizes, realDataframe['Serialmean']/realDataframe['controlmean'], color='b', xerr=[0]*len(sizes), yerr=realDataframe['Serialsem']/realDataframe['controlmean'])
    legendNames.append('Serial RAJA')
    plt.errorbar(sizes, realDataframe['controlmean']/realDataframe['controlmean'], color='g', xerr=[0]*len(sizes), yerr=realDataframe['controlsem']/realDataframe['controlmean'])
    legendNames.append('Control')
    plt.legend(legendNames, loc='center right')
    plt.xlabel('Matrix Size', fontsize=16)
    plt.ylabel('Relative Time', fontsize=16)
    plt.grid(b=True, which='major', color='k', linestyle='dotted')
    plt.title('Matrix Multiplication Speedup (24 cores)', fontsize=22)
    plt.savefig(filename)


def plot_raja_reducer(cali_file, filename, cali_query):
    ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'initialization', 'baseline', 'RajaSerial', 'OMP','AgencyParallel','AgencySerial', 'time.inclusive.duration'])
    CALI_QUERY = _get_cali_query(cali_query, ANNOTATIONS, cali_file)

    p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
    out, err = p.communicate()

    data = {
        'init': {},
        'baseline': {},
        'OMP': {},
        'RajaSerial': {},
        'AgencyParallel': {},
        'AgencySerial': {}

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
    control = data['baseline']
    Serial = data['RajaSerial']
    AgencyParallel = data['AgencyParallel']
    AgencySerial = data['AgencySerial']


    dat = [[size, c, omp, serial, agp, ags] for size in OMP for omp, serial, c, agp, ags in zip(OMP[size], Serial[size], control[size], AgencyParallel[size], AgencySerial[size])]
    dataframe = pd.DataFrame(data=dat, columns=['Size', 'baseline', 'OMP', 'RajaSerial', 'AgencyParallel', 'AgencySerial'])
    d = [[size,
          dataframe[dataframe['Size'] == size]['OMP'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['OMP']),
          dataframe[dataframe['Size'] == size]['RajaSerial'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['RajaSerial']),
          dataframe[dataframe['Size'] == size]['baseline'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['baseline']),
          dataframe[dataframe['Size'] == size]['AgencyParallel'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['AgencyParallel']),
          dataframe[dataframe['Size'] == size]['AgencySerial'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['AgencySerial'])
         ] for size in sorted(OMP)]
    realDataframe = pd.DataFrame(data=d, columns=['Size', 'OMPmean', 'OMPsem', 'RajaSerialmean', 'RajaSerialsem', 'baselinemean', 'baselinesem','AgencyParallelmean','AgencyParallelsem','AgencySerialmean', 'AgencySerialsem'])

    fig = plt.figure()
    sizes = sorted(OMP)
    legendNames = []
    plt.errorbar(sizes, realDataframe['OMPmean'], color='r', xerr=[0]*len(sizes), yerr=realDataframe['OMPsem'])
    legendNames.append('OMP RAJA')
    plt.errorbar(sizes, realDataframe['RajaSerialmean'], color='b', xerr=[0]*len(sizes), yerr=realDataframe['RajaSerialsem'])
    legendNames.append('Serial RAJA')
    plt.errorbar(sizes, realDataframe['baselinemean'], color='g', xerr=[0]*len(sizes), yerr=realDataframe['baselinesem'])
    legendNames.append('Control')
    plt.errorbar(sizes, realDataframe['AgencySerialmean'], color='c', xerr=[0]*len(sizes), yerr=realDataframe['AgencySerialsem'])
    legendNames.append('Serial Agency RAJA')
    plt.errorbar(sizes, realDataframe['AgencyParallelmean'], color='m', xerr=[0]*len(sizes), yerr=realDataframe['AgencyParallelsem'])
    legendNames.append('Parallel Agency RAJA')
    plt.legend(legendNames, loc='best', fontsize=12)
    plt.xlabel('Reduction Size', fontsize=16)
    plt.ylabel('Absolute Time', fontsize=16)
    plt.grid(b=True, which='major', color='k', linestyle='dotted')
    plt.title('Reducer Time Taken (32 cores)', fontsize=22)
    plt.yscale('log')
    plt.savefig(filename)


if __name__ == '__main__':
    main()
