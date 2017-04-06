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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--cali_file', help="The cali file to be analyzed should go here")
    parser.add_argument('parallel_type', choices=['raja', 'agency'], help="Parallelism framework being used.")
    parser.add_argument('application_type', choices=['fft', 'mmult', 'fft2d', 'mmultgpu'], help="Application benchmark being plotted")
    parser.add_argument('-o', '--output_file', help="Name of the output file")
    parser.add_argument('-q', '--cali_query_loc', help="Path to cali query executable")

    args = parser.parse_args()
    if args.parallel_type == 'raja' and args.application_type == 'mmult':
        f = plot_raja_mmult
    elif args.parallel_type == 'raja' and args.application_type == 'fft2d':
        f = plot_raja_mmult
    elif args.parallel_type == 'raja' and args.application_type == 'mmultgpu':
        f = plot_raja_mmultgpu
    else:
        sys.exit("Parallel framework {} and application {} not yet supported.".format(args.parallel_type, args.application_type))

    f(args.cali_file, args.output_file, args.cali_query_loc)

def _get_cali_query(cali_query, annotations, cali_file):
    return [cali_query, '-e', '--print-attributes={}'.format(annotations), cali_file]

def mean(it):
    it = list(it)
    return sum(it) / len(it)

def sem(it):
    it = list(it)
    return stats.sem(it)


def plot_raja_mmult(cali_file, filename, cali_query):
    ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'initialization', 'control', 'Serial', 'OMP', 'Agency', 'AgencyOMP', 'rawOMP', 'time.inclusive.duration'])
    CALI_QUERY = _get_cali_query(cali_query, ANNOTATIONS, cali_file)

    p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
    out, err = p.communicate()

    data = {
        'init': {},
        'control': {},
        'OMP': {},
        'rawOMP': {},
        'Serial': {},
        'Agency': {},
        'AgencyOMP': {}
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
    rawOMP = data['rawOMP']
    control = data['control']
    Serial = data['Serial']
    Agency = data['Agency']
    Agency_omp = data['AgencyOMP']

    dat = [[size, c, omp, serial, agency, agency_omp, raw] 
            for size in OMP 
            for omp, serial, c, agency, agency_omp, raw in zip(
                OMP[size], Serial[size], control[size], Agency[size], Agency_omp[size], rawOMP[size])
          ]
    dataframe = pd.DataFrame(data=dat, columns=['Size', 'Control', 'OMP', 'Serial', 'Agency', 'AgencyOMP', 'rawOMP'])
    d = [[size,
          dataframe[dataframe['Size'] == size]['OMP'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['OMP']),
          dataframe[dataframe['Size'] == size]['Serial'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['Serial']),
          dataframe[dataframe['Size'] == size]['Control'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['Control']),
          dataframe[dataframe['Size'] == size]['Agency'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['Agency']),
          dataframe[dataframe['Size'] == size]['AgencyOMP'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['AgencyOMP']),
          dataframe[dataframe['Size'] == size]['rawOMP'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['rawOMP']),
         ] for size in sorted(OMP)]
    realDataframe = pd.DataFrame(data=d, columns=['Size', 'OMPmean', 'OMPsem', 'Serialmean', 'Serialsem', 'controlmean', 'controlsem', 'agencymean', 'agencysem', 'agencyompmean', 'agencyompsem', 'rawmean', 'rawsem'])


    fig = plt.figure()
    sizes = sorted(OMP)
    data = [(realDataframe['OMPmean'], realDataframe['OMPsem'], 'RAJA w/ OpenMP'),
            (realDataframe['Serialmean'], realDataframe['Serialsem'], 'RAJA Serial'),
            (realDataframe['agencymean'], realDataframe['agencysem'], 'RAJA w/ Agency'),
            (realDataframe['agencyompmean'], realDataframe['agencyompsem'], 'RAJA w/ AgencyOMP'),
            (realDataframe['rawmean'], realDataframe['rawsem'], 'Raw OpenMP'),
            (realDataframe['controlmean'], realDataframe['controlsem'], 'Control')]
    colors = cm.jet(np.linspace(0, 1, len(data)))
    markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    legendNames = []
    xerr = [0] * len(sizes)
    for i, (meanData, errData, legendName) in enumerate(data):
        plt.errorbar(sizes, meanData, color=colors[i], xerr=xerr, yerr=errData, marker=markers[i])
        legendNames.append(legendName)

    plt.legend(legendNames, loc='best', fontsize=12)
    plt.xlabel('Matrix Size', fontsize=16)
    plt.ylabel('Relative Time Taken (ms)', fontsize=16)
    plt.yscale('log')
    plt.grid(b=True, which='major', color='k', linestyle='dotted')
    #plt.title('Matrix Multiplication Relative Performance (72 cores)', fontsize=22)
    plt.savefig(filename)

def plot_raja_mmultgpu(cali_file, baseFileName, cali_query, title=True, legend=True):
    ANNOTATIONS = ':'.join(['iteration', 'size', 'mode', 'time.inclusive.duration'])
    CALI_QUERY = _get_cali_query(cali_query, ANNOTATIONS, cali_file)

    p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
    out, err = p.communicate()

    data = {}

    for line in out.split():
        line = dict(map(lambda x: x.split('='), line.split(',')))
        if len(line) != 4:
            continue
        time = int(line['time.inclusive.duration'])
        size = int(line['size'])
        mode = line['mode']

        if mode not in data:
            data[mode] = {}
        if size not in data[mode]:
            data[mode][size] = []
        data[mode][size].append(time)

    colors = ('r', 'b', 'k', 'g')
    markers = ('o', 'v', 's', '*')
    namelist = (('Mmult1_cuda', 'RAJA w/ CUDA Loop Fusion'),
                ('Mmult2_cuda', 'RAJA w/ CUDA normal'),
                ('Mmult1_agency', 'RAJA w/ Agency Loop Fusion'),
                ('Mmult2_agency', 'RAJA w/ Agency normal'))
    names = { key: (name, colors[i], markers[i])
              for i, (key, name) in enumerate(namelist) }
        
    fig = plt.figure()
    legendNames = []
    for mode, datum in data.iteritems():
        sizes = sorted(datum.keys())
        xerr = [0] * len(sizes)
        means = [mean(datum[size]) for size in sizes]
        sems = [sem(datum[size]) for size in sizes]
        name, color, marker = names[mode]
        legendNames.append(name)
        plt.errorbar(sizes, means, xerr=xerr, yerr=sems, marker=marker, color=color)
    if legend:
        plt.legend(legendNames, loc='best', fontsize=12)
    if title:
        plt.title('Matrix Multiplication Performance', fontsize=22)
    plt.xlabel('Matrix Size', fontsize=16)
    plt.ylabel('Time Taken (ms)', fontsize=16)
    plt.yscale('log')
    plt.grid(b=True, which='major', color='k', linestyle='dotted')
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'figs',
        '{}_gpu.pdf'.format(baseFileName))
    plt.savefig(filename)

if __name__ == '__main__':
    main()
