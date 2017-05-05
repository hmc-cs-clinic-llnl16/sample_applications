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

here = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--cali_file', help="The cali file to be analyzed should go here")
    parser.add_argument('parallel_type', choices=['raja', 'agency'], help="Parallelism framework being used.")
    parser.add_argument('application_type', choices=['fft', 'mmult', 'fft2d', 'mmultgpu','reducer'], help="Application benchmark being plotted")
    parser.add_argument('-o', '--output_file', help="Name of the output file")
    parser.add_argument('-q', '--cali_query_loc', help="Path to cali query executable")

    functions = {
        'raja': {
            'mmult': plot_raja_mmult,
            'fft2d': lambda x, y, z : plot_raja_fft(x, y, z, "2D FFT Performance"),
            'fft': lambda x, y, z : plot_raja_fft(x, y, z, "1D FFT Performance"),
            'mmultgpu': plot_raja_mmultgpu,
            'reducer': plot_raja_reducer
        }
    }

    args = parser.parse_args()
    try:
        f = functions[args.parallel_type][args.application_type]
    except KeyError:
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

def get_cali_data(cali_query, cali_file, annotations):
    query = _get_cali_query(cali_query, annotations, cali_file)
    p = subprocess.Popen(query, stdout=subprocess.PIPE)
    return p.communicate()

def extract_cali_data(lines, keys, transformers):
    data = {}
    for line in lines:
        line = dict(map(lambda x: x.split('='), line.split(',')))
        if len(line) != len(keys) + 1:
            continue
        for key, trans in zip(keys, transformers):
            vals[key] = trans(line[key])
        last = data
        for key in keys[:-2]:
            v = vals[key]
            if v not in last:
                last[v] = {}
                last = last[v]
        list_key = vals[keys[-2]]
        if list_key not in last:
            last[list_key] = []
        last[list_key].append(vals[keys[-1]])

    return data

def plot_data(data, names, xlabel, ylabel, filename, legend=True, title=None, yscale='log', xscale=None):
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

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if yscale: plt.yscale(yscale)
    if xscale: plt.xscale(xscale)
    plt.grid(b=True, which='major', color='k', linestyle='dotted')

    plt.savefig(filename)

def noop(_): return _

def plot_raja_mmult(cali_file, baseFileName, cali_query, legend=True, title=True):
    ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'depth', 'perm', 'mode',
                            'time.inclusive.duration'])
    out, err = get_cali_data(cali_query, cali_file, ANNOTATIONS)

    keys = ('depth', 'perm', 'mode', 'size', 'time.inclusive.duration')
    trans = (noop, noop, noop, int, int)
    data = extract_cali_data(out.split(), keys, trans) 

    namelist = (('OMP', 'RAJA w/ OpenMP', 'r', 'o'),
                ('Serial', 'RAJA Serial', 'b', 'v'),
                ('Agency', 'RAJA w/ Agency', 'k', '8'),
                ('AgencyOMP', 'RAJA w/ Agency w/ OpenMP', 'g', 's'),
                ('control', 'Control Serial', 'm', '*'),
                ('rawOMP', 'Control OpenMP', 'c', 'h'))
    names = { key: (name, color, marker)
              for key, name, color, marker in namelist }

    for depth, perm_dict in data.iteritems():
        for perm, mode_dict in perm_dict.iteritems():
            filename = os.path.join(
                here,
                'figs',
                "{}_PERM_{}_Depth_{}.pdf".format(baseFileName, perm, depth))
            plot_data(mode_dict, names, 'Matrix Size', 'Time Taken (ms)', filename, title='Matrix Multiplication Performance') 

def plot_raja_fft(cali_file, baseFileName, cali_query, title, legend=True):          
    ANNOTATIONS = ':'.join(['iteration', 'size', 'mode', 'time.inclusive.duration'])
    out, err = get_cali_data(cali_query, cali_file, ANNOTATIONS)

    keys = ('mode', 'size', 'time.inclusive.duration')
    trans = (noop, int, int)
    data = extract_cali_data(out.split(), keys, trans) 

    namelist = (('OpenMP', 'RAJA w/ OpenMP', 'r', 'o'),
                ('Serial', 'RAJA Serial', 'b', 'v'),
                ('Agency', 'RAJA w/ Agency', 'k', '8'),
                ('AgencyOmp', 'RAJA w/ Agency w/ OpenMP', 'g', 's'),
                ('control', 'Control Serial', 'm', '*'))
    names = { key: (name, color, marker)
              for key, name, color, marker in namelist }

    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'figs',
        "{}.pdf".format(baseFileName))
    plot_data(data, names, 'Signal Length', 'Time Taken (ms)', filename, title=title, xscale='log')

def plot_raja_mmultgpu(cali_file, baseFileName, cali_query, title=True, legend=True):
    ANNOTATIONS = ':'.join(['iteration', 'size', 'mode', 'time.inclusive.duration'])
    out, err = get_cali_data(cali_query, cali_file, ANNOTATIONS)

    keys = ('mode', 'size', 'time.inclusive.duration')
    trans = (noop, int, int)
    data = extract_cali_data(out.split(), keys, trans) 

    namelist = (('Mmult1_cuda', 'RAJA w/ CUDA Loop Fusion', 'r', 'o'),
                ('Mmult2_cuda', 'RAJA w/ CUDA normal', 'b', 'v'),
                ('Mmult1_agency', 'RAJA w/ Agency Loop Fusion', 'k', 's'),
                ('Mmult2_agency', 'RAJA w/ Agency normal', 'g', '*'))
    names = { key: (name, color, marker)
              for key, name, color, marker in namelist }
        
    filename = os.path.join(
        here,
        'figs',
        '{}_gpu.pdf'.format(baseFileName))
    plot_data(data, names, 'Matrix Size', 'Time Taken (ms)', filename, title='Matrix Multiplication Performance')

def plot_raja_reducer(cali_file, filename, cali_query, legend=True, title=True):
    ANNOTATIONS = ':'.join(['iteration', 'loop', 'size', 'initialization', 'baseline', 'RajaSerial', 'OMP','AgencyParallel','AgencySerial','RawAgency','RawOMP' ,'time.inclusive.duration'])
    out, err = get_cali_data(cali_query, cali_file, ANNOTATIONS)

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
        if len(line) != 4:
            continue
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
                data[mode][size] = []
                
            data[mode][size].append(time)

    namelist = (('OMP', 'OMP RAJA', 'k', None),
                ('baseline', 'Control', 'r', 'dashed'),
                ('RajaSerial', 'Serial RAJA', 'r', None),
                ('AgencyParallel', 'Parallel Agency RAJA', 'b', None),
                ('AgencySerial', 'Serial Agency RAJA', 'c', None),
                ('RawAgency', 'Raw Agency', 'b', 'dashed'),
                ('RawOMP', 'Raw OMP', 'k', 'dashed'))
    names = { key : (name, color, linestyle)
              for key, name, color, linestyle in namelist }

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
      plt.legend(legendNames, loc='best', fontsize=8)
    if title:
      plt.title('Reducer Time Taken (32 cores)', fontsize=22)

    plt.xlabel('Reduction Size', fontsize=16)
    plt.ylabel('Time Taken (ms)', fontsize=16)
    plt.yscale('log')
    plt.grid(b=True, which='major', color='k', linestyle='dotted')
    filename = os.path.join(
        here,
        'figs',
        '{}.pdf'.format(filename)
    )
    plt.savefig(filename)

if __name__ == '__main__':
    main()
