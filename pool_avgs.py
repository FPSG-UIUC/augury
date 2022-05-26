# /usr/bin/env python3
""" Process many logs in parallel """

import os
import configargparse
import sys
import numpy as np
import glob

# from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial
from timeit import default_timer as timer
import pandas as pd
import re

from tqdm.contrib.concurrent import process_map

import res_vis

# from res_vis import plot_raw_hmaps, add_args


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


parser = configargparse.ArgParser()
log_parser = parser.add_argument_group("Log file processing")
log_parser.add_argument(
    "run_info", type=str, help="Name to use for run (has little/no impact)"
)

data_group = parser.add_mutually_exclusive_group(required=True)
data_group.add_argument(
    "--pattern", type=str, help="Use regex and glob to choose files"
)
data_group.add_argument(
    "--list", type=str, nargs="+", help="Take file names directly as a list"
)

log_parser.add_argument("--save_file", default="out/processed.csv", type=str)
log_parser.add_argument("--omit_outliers", action="store_true")
log_parser.add_argument("--median", action="store_true")
log_parser.add_argument("--remake", action="store_true")
log_parser.add_argument("--baseline_key", default="baseline")
log_parser.add_argument("--aop_key", default="aop")
log_parser.add_argument(
    "--outnames",
    nargs="+",
    default=["aop_", "baseline_"],
    help="File names to use instead of aop and baseline",
)
log_parser.add_argument("--fname_pattern", default="*aop_*", help="Test index pattern")
log_parser.add_argument(
    "--delimiter", default="_", help="What char to use when breaking up info"
)
log_parser.add_argument(
    "--hex", action="store_true", help="Treat indices as hex instead of int"
)
log_parser.add_argument(
    "--max_workers", default=cpu_count() * 2, help="Number of workers to use in pool"
)
log_parser.add_argument(
    "--group_by",
    default=0,
    type=int,
    help="Which entry in directory name to use as the "
    "x-value. Folder names should be formatted "
    "[run-name]_[additional-info]_[x-value]. This "
    " parameter allows for: "
    "[run-name]_[x_value]_[additional-info]. Values "
    "should increment from zero and are read "
    "right-to-left!",
)
parser.add_argument("--no_plot", action="store_true")
_, _, stat_strategies = res_vis.add_args(parser)
# assert(p is None)


def read_dat(fname_pair, args=None):
    """Get info for an aop/baseline out pair"""
    with open(fname_pair[1][0]) as _f:
        try:
            aop_addr, base_addr = _f.readline().strip().split(",")
        except ValueError as _e:
            print(f"Error in {fname_pair[1][0]}")
            raise _e
    if aop_addr == "test" or base_addr == "train":
        aop_addr = None
        base_addr = None

    dat = dict()
    for fname in fname_pair[1]:
        # if args.debug:
        #     print(f"Using {fname.split('/')[-1].split('.')[0]} as name")
        dat[fname.split("/")[-1].split(".")[0]] = pd.read_csv(
            fname, skiprows=0 if aop_addr is None else 1
        )

    res = pd.concat(dat, axis=1)
    # res = pd.concat({f'aop_{fname_pair[0]}':
    #                  pd.read_csv(aop_fname, skiprows=skiprows),
    #                  f'baseline_{fname_pair[0]}':
    #                  pd.read_csv(bas_fname, skiprows=skiprows)},
    #                 axis=1)

    # get names of runs with no data
    # z_runs = res[res != 0].count() == 0
    # z_runs = res.keys()[z_runs]
    # if len(z_runs) > 0:
    #     z_str = ', '.join([f'{x[0]}:{x[1]}' for x in z_runs])
    #     print(f'{z_str} have no data')

    # drop runs with no data
    # nz_runs = res[res != 0].count() != 0
    # short_res = res[res.keys()[nz_runs]]

    # count zero values across all runs
    # count = short_res.count().sum()
    # zeroes = (short_res == 0).sum().sum()
    # print(f'Omitted {zeroes} zero values / {count} values')

    # drop zero values
    res = res[res != 0]

    if args.omit_outliers:
        # omitted = res.count().sum() - (res < res.mean() +
        #                                2*res.std()).sum().sum()
        # print(f'Additionally omitted {omitted} outliers / '
        #       f'{count - zeroes} remaining values')
        res = res[res < res.mean() + 2 * res.std()]

    try:
        keys = {
            "train_base_key": [
                k[0] for k in res.keys() if "baseline" in k[0] and "train" in k[0]
            ][0],
            "train_aop_key": [
                k[0] for k in res.keys() if "aop" in k[0] and "train" in k[0]
            ][0],
            "test_base_key": [
                k[0] for k in res.keys() if "baseline" in k[0] and "test" in k[0]
            ][0],
            "test_aop_key": [
                k[0] for k in res.keys() if "aop" in k[0] and "test" in k[0]
            ][0],
        }
        assert len(fname_pair[1]) == 4, "Wrong number of files"
    except IndexError:
        keys = {
            # 'train_base_key': [k[0] for k in res.keys() if 'ab_data' in
            #                    k[0]][0],
            # 'train_aop_key': [k[0] for k in res.keys() if 'a_data' in k[0]][0],
            # 'test_base_key': [k[0] for k in res.keys() if 'ab_data' in
            #                   k[0]][0],
            # 'test_aop_key': [k[0] for k in res.keys() if 'a_dat' in k[0]][0]
            "train_base_key": [k[0] for k in res.keys() if args.baseline_key in k[0]][
                0
            ],
            "train_aop_key": [k[0] for k in res.keys() if args.aop_key in k[0]][0],
            "test_base_key": [k[0] for k in res.keys() if args.baseline_key in k[0]][0],
            "test_aop_key": [k[0] for k in res.keys() if args.aop_key in k[0]][0],
        }
        assert len(fname_pair[1]) == 2, "Wrong number of files"

    # print(train_base_key, addrs['base'])
    # # print(final_res[train_base_key].agg([np.mean, np.std]))
    # print(res[train_base_key][['test', 'train']].agg([np.mean, np.std]))
    # print('')
    # print(train_aop_key, addrs['aop'])
    # # print(final_res[train_aop_key].agg([np.mean, np.std]))
    # print(res[train_aop_key][['test', 'train']].agg([np.mean, np.std]))

    if not args.median:
        test_base = (
            res.mean()[:, "test"][keys["test_base_key"]],
            res.std()[:, "test"][keys["test_base_key"]],
        )
        test_aop = (
            res.mean()[:, "test"][keys["test_aop_key"]],
            res.std()[:, "test"][keys["test_aop_key"]],
        )
        train_base = (
            res.mean()[:, "train"][keys["train_base_key"]],
            res.std()[:, "train"][keys["train_base_key"]],
        )
        train_aop = (
            res.mean()[:, "train"][keys["train_aop_key"]],
            res.std()[:, "train"][keys["train_aop_key"]],
        )
    else:
        test_base = (
            res.median()[:, "test"][keys["test_base_key"]],
            res.std()[:, "test"][keys["test_base_key"]],
        )
        test_aop = (
            res.median()[:, "test"][keys["test_aop_key"]],
            res.std()[:, "test"][keys["test_aop_key"]],
        )
        train_base = (
            res.median()[:, "train"][keys["train_base_key"]],
            res.std()[:, "train"][keys["train_base_key"]],
        )
        train_aop = (
            res.median()[:, "train"][keys["train_aop_key"]],
            res.std()[:, "train"][keys["train_aop_key"]],
        )

    test_speedup = test_base[0] / test_aop[0]
    train_speedup = train_base[0] / train_aop[0]

    # print(bcolors.OKBLUE)
    # print(f'{args.run_info} Test  Speedup: {test_speedup:.3f} '
    #       f'({test_base[0]:.3f}±{test_base[1]:.2f}/'
    #       f'{test_aop[0]:.3f}±{test_aop[1]:.2f})')
    # print(f'{args.run_info} Train Speedup: {train_speedup:.3f} '
    #       f'({train_base[0]:.3f}±{train_base[1]:.2f}/'
    #       f'{train_aop[0]:.3f}±{train_aop[1]:.2f})')
    # print(bcolors.ENDC)

    info_str = (
        f"{fname_pair[2]}_{fname_pair[0]},"
        f"{train_speedup:3f}"
        f",{test_speedup:3f},"
        f"{aop_addr},"
        f"{base_addr},"
        f"{train_aop[0]:.4f},{train_aop[1]:.4f},"
        f"{train_base[0]:.4f},{train_base[1]:.4f},"
        f"{test_aop[0]:.4f},{test_aop[1]:.4f},"
        f"{test_base[0]:.4f},{test_base[1]:.4f}"
    )

    # return info_str, test_loc, (aop_addr, base_addr), res
    return info_str


def make_pair(test_loc: int, dirname: str, info: str, out_fname_list):
    """Use test loc to build a filename"""
    fnames = []
    if len(out_fname_list) == 2:
        for _f in out_fname_list:
            fname = f"{dirname}/{_f}{test_loc}.out"
            if not os.path.isfile(fname):
                print(f"Failed to find {fname}")
                return (test_loc, None, None, None)
            fnames.append(fname)

        return (test_loc, fnames, info)

    # # Check for single-process results
    # aop_fname = f'{dirname}/aop_{test_loc}.out'
    # bas_fname = f'{dirname}/baseline_{test_loc}.out'
    # if os.path.isfile(aop_fname) and os.path.isfile(bas_fname):
    #     # add print here --> this is the case Michael's logs should be falling
    #     # into
    #     return (test_loc, [aop_fname, bas_fname], info)

    # Check for multi-process results
    if len(out_fname_list) == 4:
        aop_train_fname = f"{dirname}/aop_train_{test_loc}.out"
        bas_train_fname = f"{dirname}/baseline_train_{test_loc}.out"
        aop_test_fname = f"{dirname}/aop_test_{test_loc}.out"
        bas_test_fname = f"{dirname}/baseline_test_{test_loc}.out"

    if (
        os.path.isfile(aop_train_fname)
        and os.path.isfile(bas_train_fname)
        and os.path.isfile(aop_test_fname)
        and os.path.isfile(bas_test_fname)
    ):
        return (
            test_loc,
            [aop_train_fname, aop_test_fname, bas_train_fname, bas_test_fname],
            info,
        )

    # Found no results!
    # print(f'Found nothing for {dirname}/')
    return (test_loc, None, None, None)


def read_testsweep(
    dirname: str, group_by: int, fname_pattern: str, delimiter: str, outname_list
):
    """Read a single directory of test index sweep values."""
    fnames = glob.glob(f"{dirname}/{fname_pattern}.out")
    if len(fnames) == 0:
        print(f"{dirname} is empty, skipping")
        return []

    if args.hex:
        test_locs = np.array(
            [int(f.split("/")[-1].split(".")[0].split("_")[-1], 16) for f in fnames]
        )
    else:
        test_locs = np.array(
            [f.split("/")[-1].split(".")[0].split("_")[-1] for f in fnames]
        )
    # if args.debug:
    #     print(f'Found {test_locs} in {dirname}')

    if not args.hex:
        test_locs = test_locs.astype(int)

    test_locs.sort()

    if args.hex:
        test_locs = [hex(t) for t in test_locs]

    test_locs = np.unique(test_locs)

    if args.debug:
        print(f"Found {test_locs} in {dirname}")

    _info = dirname.split("/")[-1].split(delimiter)[group_by]
    # _info = dirname.split('/')[-1].split(delimiter)[0]

    _info = _info.replace("(TM)", "")
    _info = _info.replace("(R)", "")
    _info = _info.replace("@", "")
    _info = _info.replace("(", "")
    _info = _info.replace(")", "")

    _info = f'{args.run_info.replace("_", "-")}_{_info.replace("_", "-")}'

    if args.debug:
        print(
            f"Info is:"
            f" {dirname}"
            f' --> {dirname.split("/")[-1]}'
            f' --> {dirname.split("/")[-1].split("_")[group_by]}'
            f" --> {_info}"
        )

    fname_pairs = [make_pair(t, dirname, _info, outname_list) for t in test_locs]
    # remove invalid configs
    fname_pairs = [pair for pair in fname_pairs if pair[1] is not None]

    return fname_pairs


def read_sweep(args):
    """Read an entire sweep"""
    if args.pattern is not None:
        dirnames = glob.glob(args.pattern)
    else:
        dirnames = args.list

    # path = '/'.join(pattern.split('/')[:-1])
    # ds = list(filter(re.compile(pattern.split('/')[-1]).match,
    #                        os.listdir(path)))
    # dirnames = [path + '/' + d for d in ds]

    dirnames.sort()
    if args.debug:
        print(f'Found: {", ".join(dirnames[0:5])}, ... ,' f'{", ".join(dirnames[-5:])}')

    all_dirs = list()
    for dirname in dirnames:
        all_dirs += read_testsweep(
            dirname,
            -1 * (args.group_by),
            args.fname_pattern,
            args.delimiter,
            args.outnames,
        )

    if args.debug:
        # print(f'Found: {", ".join(all_dirs[0:5])}, ... ,'
        #       f'{", ".join(all_dirs[-5:])}')
        print(f"Found: {all_dirs[0:5]}")

    # with Pool(32) as _p:
    #     all_dat = _p.map(read_dat, all_dirs)
    read_func = partial(read_dat, args=args)
    if args.debug:
        print(f"Using {args.max_workers} workers")
    all_dat = process_map(
        read_func, all_dirs, max_workers=args.max_workers, unit="configs", chunksize=1
    )

    return all_dirs, all_dat


def process_logs(args):
    start = timer()
    all_names, all_res = read_sweep(args)
    start_write = timer()

    assert len(all_res) > 0, "Nothing loaded?"

    with open(args.save_file, "w+") as _f:
        _f.write(
            "name,train,test,aop_addr,base_addr,"
            "train_aop,train_aop_std,train_base,train_base_std,"
            "test_aop,test_aop_std,test_base,test_base_std"
            "\n"
        )
        _f.write("\n".join(all_res))
    end = timer()

    print(
        f"Processed {len(all_names)} configurations in "
        f"{start_write - start:.4f} seconds"
    )

    if end - start_write > 1.0:
        print(f"Wrote in {end - start_write:.4f} seconds")

    return all_names


if __name__ == "__main__":
    args = parser.parse_args()
    # print(args)
    # print('')

    load_files = True
    if os.path.isfile(args.save_file):
        # if save file exists, prompt to skip
        load_files = (
            args.remake
            or (input(f"{args.save_file} exists, re-make it? [y/N] ") or "n").lower()[0]
            == "y"
        )

    if load_files:
        processed_files = process_logs(args)
    else:
        print(f"Left existing {args.save_file}")
        assert os.path.isfile(args.save_file), "How did you get here..?"

    if args.no_plot:
        sys.exit(0)

    dat = res_vis.load_data(args, fname=args.save_file)

    # dat = pd.read_csv(args.save_file).set_index('name')
    # dat = dat.sort_index(key=lambda col: col.map(lambda x:
    #                                              int(x.split('_')[1])*100 +
    #                                              int(x.split('_')[2])))

    _, _, raw_dat = res_vis.filter_data(dat, stat_strategies, args)
    EXP_FNAME = "_".join(args.save_file.split(".")[:-1])
    rc = {"figure.figsize": (15, 8)}
    res_vis.plot_raw_hmaps(raw_dat, EXP_FNAME, "test", args, rc)
