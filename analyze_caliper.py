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
    parser.add_argument('application_type', choices=['fft', 'mmult', 'fft2d', 'mmultgpu','reducer'], help="Application benchmark being plotted")
    parser.add_argument('-o', '--output_file', help="Name of the output file")
    parser.add_argument('-q', '--cali_query_loc', help="Path to cali query executable")

    args = parser.parse_args()
    if args.parallel_type == 'raja' and args.application_type == 'mmult':
        f = plot_raja_mmult
    elif args.parallel_type == 'raja' and args.application_type == 'fft2d':
        f = lambda x, y, z : plot_raja_fft(x, y, z, "2D FFT Performance")
    elif args.parallel_type == 'raja' and args.application_type == 'reducer':
        f = plot_raja_reducer
    elif args.parallel_type == 'raja' and args.application_type == 'fft':
        f = lambda x, y, z : plot_raja_fft(x, y, z, "1D FFT Performance")
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

def plot_raja_mmult(cali_file, baseFileName, cali_query, legend=True, title=True):
    ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'depth', 'perm', 'mode',
                            'time.inclusive.duration'])
    CALI_QUERY = _get_cali_query(cali_query, ANNOTATIONS, cali_file)

    p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
    out, err = p.communicate()

    data = {}

    for line in out.split():
        line = dict(map(lambda x: x.split('='), line.split(',')))
        if len(line) != 6:
            continue

        depth = line['depth']
        mode = line['mode']
        perm = line['perm']
        size = int(line['size'])
        time = int(line['time.inclusive.duration'])

        if depth and perm and mode:
            if depth not in data:
                data[depth] = {}
            if perm not in data[depth]:
                data[depth][perm] = {}
            if mode not in data[depth][perm]:
                data[depth][perm][mode] = {}
            if size not in data[depth][perm][mode]:
                data[depth][perm][mode][size] = []
            data[depth][perm][mode][size].append(time)

    colors = ['r', 'b', 'k', 'g', 'c', 'm']
    markers = ('o', 'v', '8', 's', 'h', '*')
    namelist = (('OMP', 'RAJA w/ OpenMP'),
                ('Serial', 'RAJA Serial'),
                ('Agency', 'RAJA w/ Agency'),
                ('AgencyOMP', 'RAJA w/ Agency w/ OpenMP'),
                ('control', 'Control Serial'),
                ('rawOMP', 'Control OpenMP'))
    names = { key: (name, colors[i], markers[i]) 
              for i, (key, name) in enumerate(namelist) }

    for depth, perm_dict in data.iteritems():
        for perm, mode_dict in perm_dict.iteritems():
            fig = plt.figure()
            legendNames = []
            for i, (mode, datum) in enumerate(mode_dict.iteritems()):
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
                "{}_PERM_{}_Depth_{}.pdf".format(baseFileName, perm, depth))
            plt.savefig(filename)

def plot_raja_fft(cali_file, baseFileName, cali_query, title, legend=True):          
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
        mode = line['mode']
        size = int(line['size'])

        if mode not in data:
            data[mode] = {}
        if size not in data[mode]:
            data[mode][size] = []

        data[mode][size].append(time)

    colors = ['r', 'b', 'k', 'g', 'c']
    markers = ('o', 'v', 's', 'h', '*')
    namelist = (('OpenMP', 'RAJA w/ OpenMP'),
                ('Serial', 'RAJA Serial'),
                ('Agency', 'RAJA w/ Agency'),
                ('AgencyOmp', 'RAJA w/ Agency w/ OpenMP'),
                ('control', 'Control Serial'))
    names = { key: (name, colors[i], markers[i]) 
              for i, (key, name) in enumerate(namelist) }

    fig = plt.figure()
    legendNames = []
    for i, (mode, datum) in enumerate(data.iteritems()):
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
      plt.title(title, fontsize=22)
    plt.xlabel('Signal Length', fontsize=16)
    plt.ylabel('Time Taken (ms)', fontsize=16)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(b=True, which='major', color='k', linestyle='dotted')
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'figs',
        "{}.pdf".format(baseFileName))
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

def plot_raja_reducer(cali_file, filename, cali_query):
    ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'initialization', 'baseline', 'RajaSerial', 'OMP','AgencyParallel','AgencySerial','RawAgency','RawOMP' ,'time.inclusive.duration'])
    CALI_QUERY = _get_cali_query(cali_query, ANNOTATIONS, cali_file)

    p = subprocess.Popen(CALI_QUERY, stdout=subprocess.PIPE)
    out, err = p.communicate()

    data = {
        'init': {},
        'baseline': {},
        'OMP': {},
        'RajaSerial': {},
        'AgencyParallel': {},
        'AgencySerial': {},
        'RawAgency' :{},
        'RawOMP' :{}

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
    RawAgency = data['RawAgency']
    RawOMP = data['RawOMP']


    dat = [[size, c, omp, serial, agp, ags, ra, ro] for size in OMP for omp, serial, c, agp, ags, ra, ro in zip(OMP[size], Serial[size], control[size], AgencyParallel[size], AgencySerial[size], RawAgency[size], RawOMP[size])]
    dataframe = pd.DataFrame(data=dat, columns=['Size', 'baseline', 'OMP', 'RajaSerial', 'AgencyParallel', 'AgencySerial','RawAgency','RawOMP'])
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
          stats.sem(dataframe[dataframe['Size'] == size]['AgencySerial']),
          dataframe[dataframe['Size'] == size]['RawAgency'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['RawAgency']),
          dataframe[dataframe['Size'] == size]['RawOMP'].mean(),
          stats.sem(dataframe[dataframe['Size'] == size]['RawOMP'])
         ] for size in sorted(OMP)]
    realDataframe = pd.DataFrame(data=d, columns=['Size', 'OMPmean', 'OMPsem', 'RajaSerialmean', 'RajaSerialsem', 'baselinemean', 'baselinesem','AgencyParallelmean','AgencyParallelsem','AgencySerialmean', 'AgencySerialsem','RawAgencymean', 'RawAgencysem','RawOMPmean', 'RawOMPsem'])

    fig = plt.figure()
    sizes = sorted(OMP)
    legendNames = []
    plt.errorbar(sizes, realDataframe['OMPmean'], color='k', xerr=[0]*len(sizes), yerr=realDataframe['OMPsem'])
    legendNames.append('OMP RAJA')
    plt.errorbar(sizes, realDataframe['RawOMPmean'], color='k', xerr=[0]*len(sizes), yerr=realDataframe['RawOMPsem'], linestyle='dashed')
    legendNames.append('Raw OMP')
    plt.errorbar(sizes, realDataframe['RajaSerialmean'], color='r', xerr=[0]*len(sizes), yerr=realDataframe['RajaSerialsem'])
    legendNames.append('Serial RAJA')
    plt.errorbar(sizes, realDataframe['baselinemean'], color='r', xerr=[0]*len(sizes), yerr=realDataframe['baselinesem'], linestyle='dashed')
    legendNames.append('Control')
    plt.errorbar(sizes, realDataframe['AgencySerialmean'], color='c', xerr=[0]*len(sizes), yerr=realDataframe['AgencySerialsem'])
    legendNames.append('Serial Agency RAJA')
    plt.errorbar(sizes, realDataframe['AgencyParallelmean'], color='b', xerr=[0]*len(sizes), yerr=realDataframe['AgencyParallelsem'])
    legendNames.append('Parallel Agency RAJA')
    plt.errorbar(sizes, realDataframe['RawAgencymean'], color='b', xerr=[0]*len(sizes), yerr=realDataframe['RawAgencysem'], linestyle='dashed')
    legendNames.append('Raw Agency')

    plt.legend(legendNames, loc='best', fontsize=8)
    plt.xlabel('Reduction Size', fontsize=16)
    plt.ylabel('Absolute Time', fontsize=16)
    plt.grid(b=True, which='major', color='k', linestyle='dotted')
    plt.title('Reducer Time Taken (32 cores)', fontsize=22)
    plt.yscale('log')
    plt.savefig(filename)


if __name__ == '__main__':
    main()
