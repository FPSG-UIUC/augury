#! /usr/bin/env python3

import os
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from itertools import permutations
from random import sample


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


parser = argparse.ArgumentParser()
parser.add_argument('fname', help='Log file to process', metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
parser.add_argument('--out_name', default='out/addr', type=str)
parser.add_argument('--no_vis', action='store_true')
args = parser.parse_args()

dat = pd.read_csv(args.fname, usecols=[1,2,3,4])  # read and omit name
# amount = len(dat) * 15
# colors = [[float(i)/float(amount), float(i)/float(amount),
#            float(amount-i)/float(amount)] for i in range(0, amount, 15)]

# Nlines = 200
color_lvl = 8
rgb = np.array(list(permutations(range(0,256,color_lvl),3)))/255.0
colors = sample(list(rgb),len(dat))

dat['colors'] = colors

by_aop = dat[['train','test','aop_addr', 'colors']]
by_dat = dat[['train','test','base_addr', 'colors']]

by_aop = by_aop.set_index('aop_addr').sort_index()
by_dat = by_dat.rename(columns={'base_addr':'dat_addr'})
by_dat = by_dat.set_index('dat_addr').sort_index()

fig, axs = plt.subplots(2, 1)
fig.set_size_inches(20, 10)
fig.canvas.manager.set_window_title(args.out_name)

axs[0].set_title('Speedup vs AOP Address')
axs[0].set_ylabel('Speedup (AOP / Base)')
axs[0].set_xlabel('AOP Address')
axs[0].tick_params(axis='x', rotation=90)
axs[1].set_title('Speedup vs Data Address')
axs[1].set_ylabel('Speedup (AOP / Base)')
axs[1].set_xlabel('Data Address')
axs[1].tick_params(axis='x', rotation=90)

axs[0].grid(linestyle='--', linewidth=0.5)
axs[0].plot([-0.5, len(set(by_aop.index))-0.5], [1,1], linestyle='--')
axs[0].set_xlim([-0.5,len(set(by_dat.index))-0.5])
axs[1].grid(linestyle='--', linewidth=0.5)
axs[1].plot([-0.5, len(set(by_dat.index))-0.5], [1,1], linestyle='--')
axs[1].set_xlim([-0.5,len(set(by_dat.index))-0.5])

axs[0].scatter(by_aop.index, by_aop['train'], color=by_aop['colors'])
axs[1].scatter(by_dat.index, by_dat['train'], color=by_dat['colors'])

plt.tight_layout()

plt.savefig(f'{args.out_name}.png')

# use this for x-axis locations -- ensures that entries "line up" to the actual
# address hex value
by_aop['xlocs'] = [int(v, 16) for v in by_aop.index]
by_dat['xlocs'] = [int(v, 16) for v in by_dat.index]

# exported to simplify plotting in veusz
by_aop.to_csv(f'{args.out_name}_aop.csv')
by_dat.to_csv(f'{args.out_name}_dat.csv')

if not args.no_vis:
    plt.show()
