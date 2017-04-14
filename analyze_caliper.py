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
    parser.add_argument('application_type', choices=['fft', 'mmult', 'fft2d'], help="Application benchmark being plotted")
    parser.add_argument('-o', '--output_file', help="Name of the output file")
    parser.add_argument('-q', '--cali_query_loc', help="Path to cali query executable")

    args = parser.parse_args()
    if args.parallel_type == 'raja' and args.application_type == 'mmult':
        plot_raja_mmult(args.cali_file, args.output_file, args.cali_query_loc)
    elif args.parallel_type == 'raja' and args.application_type == 'fft2d':
        plot_raja_mmult(args.cali_file, args.output_file, args.cali_query_loc)
    else:
        sys.exit("Parallel framework {} and application {} not yet supported.".format(args.parallel_type, args.application_type))


def mean(it):
    it = list(it)
    return sum(it) / len(it)

def sem(it):
    it = list(it)
    return stats.sem(it) 

def _get_cali_query(cali_query, annotations, cali_file):
    return [cali_query, '-e', '--print-attributes={}'.format(annotations), cali_file]


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


if __name__ == '__main__':
    main()
