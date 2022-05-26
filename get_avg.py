#!/usr/bin/env python3

import numpy as np

import sys
import os

import argparse

try:
    import pandas as pd
    imported_pandas = True

except ImportError:
    import csv
    import statistics
    import math
    print("Failed to import pandas.")
    imported_pandas = False


    class fake_DataFrame:
        def __init__(self, data: dict):
            assert(len(data) ==  2), "Too many files"
            self.data = data
            self.means = None
            return


        def count(self):
            counts = list()
            for k in self.data:
                counts.append(len(self.data[k]) * 2)
            return counts


        def mean(self):
            if self.means is None:
                means = dict()
                for k in self.data:
                    means[k] = {'test':  statistics.mean(self.data[k]['test']),
                                'train': statistics.mean(self.data[k]['train'])}
                self.means = means
            return self


        def __str__(self):
            retval = ''
            maxlen = max([len(k) for k in self.data])
            for k in self.data:
                retval += f'{k.ljust(maxlen, " ")} '
                retval += f'Test  {self.means[k]["test"]}\n'
                retval += f'{"".ljust(maxlen, " ")} '
                retval += f'Train {self.means[k]["train"]}\n\n'

            return retval


        def keys(self):
            return [(k, 'nothing') for k in self.data]


        def __getitem__(self, item):
            if type(item[0]) == slice:
                return {k: self.means[k][item[1]] for k in self.means}



    class fake_pandas:
        def __init__(self):
            self.data = None

        def read_csv(self, fname):
            test = []
            train = []
            with open(fname, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    test.append(int(row['test']))
                    train.append(int(row['train']))
            return {'test': test, 'train': train}

        def concat(self, data: dict, axis: int) -> fake_DataFrame:
            return fake_DataFrame(data)


    pd = fake_pandas()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# 1-column file -> one array of int
def parse_file_1c(fn):
    with open(fn) as f:
        lines = [int(line.strip()) for line in f]
    return np.array(lines)


# 2-column file -> two arrays of int
def parse_file_2c(fn):
    first = []
    second = []
    with open(fn) as f:
        for line in f:
            a, b = line.strip().split()
            first.append(int(a))
            second.append(int(b))
    return np.array(first), np.array(second)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', help='Log file to process', nargs='+')
    parser.add_argument('run_info', type=str)
    parser.add_argument('--save_file', default='out/processed.csv', type=str)
    parser.add_argument('--append_save', action='store_true')
    parser.add_argument('--omit_outliers', action='store_true')
    parser.add_argument('--extra_info', nargs='+')
    parser.add_argument('--save_anyway', action='store_true')
    parser.add_argument('--min_speedup', default=2.0, type=float)
    parser.add_argument('--print_counts', action='store_true')
    args = parser.parse_args()

    all_dat = dict()
    addrs = dict()
    for arg in args.fnames:
        filename = arg
        data_type = filename.split("/")[-1].split(".")[0]
        skiprows = 1
        with open(filename) as f:
            aop_addr, base_addr = f.readline().strip().split(',')
        if aop_addr == 'test' or base_addr == 'train':
            skiprows = 0
            addrs[data_type] = {'aop': None, 'base': None}
        else:
            addrs[data_type] = {'aop': aop_addr, 'base': base_addr}
        all_dat[data_type] = pd.read_csv(arg, skiprows=skiprows)

    final_res = pd.concat(all_dat, axis=1)

    z_runs = final_res[final_res != 0].count() == 0
    z_runs = final_res.keys()[z_runs]
    if len(z_runs) > 0:
        z_str = ', '.join([f'{x[0]}:{x[1]}' for x in z_runs])
        print(f'{z_str} have no data')

    nz_runs = final_res[final_res != 0].count() != 0
    short_res = final_res[final_res.keys()[nz_runs]]

    count = short_res.count().sum()
    zeroes = (short_res == 0).sum().sum()
    print(f'Omitted {zeroes} zero values / {count} values')
    if args.print_counts:
        print(final_res.count())
    final_res = final_res[final_res != 0]
    if args.print_counts:
        print(final_res.count())

    if args.omit_outliers and imported_pandas==True:
        omitted = final_res.count().sum() - (final_res < final_res.mean() +
                           2*final_res.std()).sum().sum()
        print(f'Additionally omitted {omitted} outliers / '
              f'{count - zeroes} remaining values')
        final_res = final_res[final_res < final_res.mean() + 2*final_res.std()]

    elif args.omit_outliers and imported_pandas==False:
        print("Omit outliers requires pandas")

    print(f'Using {final_res.count().sum()}\n')

    # print(final_res.mean())
    # print(final_res.std())

    try:
        train_base_key = [k[0] for k in final_res.keys() if 'baseline' in k[0]
                          and 'train' in k[0]][0]
        train_aop_key = [k[0] for k in final_res.keys() if 'aop' in k[0] and
                         'train' in k[0]][0]
        test_base_key = [k[0] for k in final_res.keys() if 'baseline' in k[0]
                         and 'test' in k[0]][0]
        test_aop_key = [k[0] for k in final_res.keys() if 'aop' in k[0] and
                        'test' in k[0]][0]
    except IndexError:
        train_base_key = [k[0] for k in final_res.keys() if 'baseline' in
                          k[0]][0]
        train_aop_key = [k[0] for k in final_res.keys() if 'aop' in k[0]][0]
        test_base_key = [k[0] for k in final_res.keys() if 'baseline' in
                         k[0]][0]
        test_aop_key = [k[0] for k in final_res.keys() if 'aop' in k[0]][0]

    print(train_base_key, addrs[train_base_key])
    # print(final_res[train_base_key].agg([np.mean, np.std]))
    print(final_res[train_base_key][['test','train']].agg([np.mean, np.std]))
    print('')
    print(train_aop_key, addrs[train_aop_key])
    # print(final_res[train_aop_key].agg([np.mean, np.std]))
    print(final_res[train_aop_key][['test','train']].agg([np.mean, np.std]))

    # print(final_res.mean()[:,'test'][test_base_key])
    # print(final_res.mean()[:,'test'][test_aop_key])
    # print(final_res.mean()[:,'train'][train_base_key])
    # print(final_res.mean()[:,'train'][train_aop_key])

    test_base = (final_res.mean()[:, 'test'][test_base_key],
                 final_res.std()[:, 'test'][test_base_key])
    test_aop = (final_res.mean()[:, 'test'][test_aop_key],
                final_res.std()[:,  'test'][test_aop_key])
    train_base = (final_res.mean()[:, 'train'][train_base_key],
                  final_res.std()[:, 'train'][train_base_key])
    train_aop = (final_res.mean()[:, 'train'][train_aop_key],
                 final_res.std()[:,  'train'][train_aop_key])

    test_speedup = test_base[0] / test_aop[0]
    train_speedup = train_base[0] / train_aop[0]

    print(bcolors.OKBLUE)
    print(f'{args.run_info} Test  Speedup: {test_speedup:.3f} '
          f'({test_base[0]:.3f}±{test_base[1]:.2f}/'
          f'{test_aop[0]:.3f}±{test_aop[1]:.2f})')
    print(f'{args.run_info} Train Speedup: {train_speedup:.3f} '
          f'({train_base[0]:.3f}±{train_base[1]:.2f}/'
          f'{train_aop[0]:.3f}±{train_aop[1]:.2f})')

    if train_speedup < args.min_speedup and not args.save_anyway:
        print(bcolors.FAIL + 'Insufficient train speedup, was the AOP on?')
        print(bcolors.ENDC)
        sys.exit(1)  # signal failure to script, experiment will repeat
    else:
        print(bcolors.ENDC)
        write_header = True
        if args.append_save and os.path.isfile(args.save_file):
            write_header = False

        print(f'Saving to {args.save_file}')
        with open(args.save_file, 'a+' if args.append_save else 'w+') as f:
            if write_header:
                f.write('name,train,test,aop_addr,base_addr,'
                        f'train_aop,train_aop_std,train_base,train_base_std,'
                        f'test_aop,test_aop_std,test_base,test_base_std,'
                        f'\n')
            f.write(f'{args.run_info},{train_speedup:3f},{test_speedup:3f},'
                    f'{addrs[train_aop_key]["aop"]},'
                    f'{addrs[train_aop_key]["base"]},'
                    f'{train_aop[0]:.4f},{train_aop[1]:.4f},'
                    f'{train_base[0]:.4f},{train_base[1]:.4f},'
                    f'{test_aop[0]:.4f},{test_aop[1]:.4f},'
                    f'{test_base[0]:.4f},{test_base[1]:.4f}\n'
                   )

        sys.exit(0)
