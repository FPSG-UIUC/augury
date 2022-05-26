#! /usr/bin/env python3

import os
import itertools
import sys

# import argparse
import configargparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import seaborn as sns


DEFAULT_YLABEL = "Distance from Train"
DEFAULT_NORMHMAP_YLABEL = "Test Index"


y_labels = {
    "speedup": "Speedup (aop / base): Higher is better",
    "raw": "Thread Ticks: Smaller is better",
    "delta": "Tick Delta (base - aop): Larger is better",
}

marker_colors = {
    "aop": (31, 119, 180),
    "base": (255, 127, 14),
    "aop_special": (44, 160, 44),
    "base_special": (214, 39, 40),
    "test": (148, 103, 189),
    "train": (140, 86, 75),
}


# def to_1D(series):
#     return pd.Series([x for _list in series for x in _list])


def load_data(args, fname=None) -> pd.DataFrame:
    """Load data into pd.dataframe"""
    if fname is None:
        dat = pd.read_csv(args.fname).set_index("name")
    else:
        dat = pd.read_csv(fname).set_index("name")

    if not args.no_sort:
        dat = dat.sort_index(
            key=lambda col: col.map(
                lambda x: int(x.split("_")[1]) * 100 + int(x.split("_")[2])
            )
        )

    dat = setup_color(dat, args)

    return dat


def get_color(name: str):
    """Convert a color name to RGB values"""
    _r, _g, _b = marker_colors.get(name, lambda: (0, 0, 0))
    return (_r / 255, _g / 255, _b / 255)


def is_valid_file(parser, arg):
    """For use with argparse. Ensure the passed input file exists"""
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    # else:
    return arg


def count_exps(dat: pd.DataFrame):
    """Given a dataframe, determine the number of different experiments which
    make it up."""
    test_indices = [x.split("_")[:-1] for x in dat.index]
    desc = dat.index[0].split("_")[0]  # run_id in test_sweep.sh
    length = list()
    previous = 0
    for i in range(len(test_indices) - 1):
        # if test_indices[i] > test_indices[i+1]: # cur > next, must have reset
        if test_indices[i] != test_indices[i + 1]:
            length.append((previous, i + 1))
            previous = i + 1

    length.append((previous, len(test_indices)))

    return len(length), length, desc


def plot_data(dat, exp_info, plot_type, args, y_label, fname):
    """Generate various types of plots for a given dataframe"""
    if not args.plot_all or (not args.visualize and args.no_save):
        return
    num_exps, exp_len, _ = exp_info
    exps_per_graph = [args.max_exps] * (num_exps // args.max_exps) + [
        num_exps % args.max_exps
    ]
    idx_iter = itertools.chain.from_iterable(range(e) for e in exps_per_graph)
    exps_per_graph = iter(exps_per_graph + [0])

    if not args.no_save:
        pdf_fname = f"{fname}_{plot_type}.pdf"
        pdf = PdfPages(pdf_fname)
        print(f"Saving to {pdf_fname}")

    fig = None

    for idx, (start, end) in tqdm(
        zip(idx_iter, exp_len), unit="experiment", total=len(exp_len)
    ):
        if idx == 0:
            if fig is not None:
                plt.tight_layout()
                if not args.no_save:
                    pdf.savefig(fig)
                plt.close()

            max_graphs = next(exps_per_graph)

            if max_graphs != 0:
                tqdm.write(f"Moving to new figure with {max_graphs} graphs")
                fig, axs = plt.subplots(2, max_graphs)
                fig.set_size_inches(8 * max_graphs, 10)

            if max_graphs == 1:
                axs = [[axs[0]], [axs[1]]]

        indices = [i.split("_")[-1] for i in dat["test"][start:end].index]

        try:
            title = "_".join(dat["train"][start:end].index[0].split("_")[:-1])
        except IndexError:
            tqdm.write(f"Empty experiment {start}:{end}")
            continue
        axs[0][idx].set_title(f"Train {title}")
        axs[0][idx].set_ylabel(f"{y_label}")
        axs[0][idx].set_xlabel(args.x_label)
        axs[0][idx].set_xlim([-0.5, args.max_x or len(indices) - 0.5])
        axs[1][idx].set_title(f"Test {title}")
        axs[1][idx].set_ylabel(f"{y_label}")
        axs[1][idx].set_xlabel("Distance from train")
        axs[1][idx].set_xlim([-0.5, args.max_x or len(indices) - 0.5])

        if plot_type.split("_")[0] == "speedup":
            axs[0][idx].set_ylim([0, dat["train"].max() * 1.1])
            axs[1][idx].set_ylim([0, dat["test"].max() + 0.5])
        elif plot_type.split("_")[0] == "raw":
            # axs[0][idx].set_ylim([0, dat[['train_aop',
            #                               'train_base']].max().max()])
            axs[1][idx].set_ylim([0, dat[["test_aop", "test_base"]].max().max()])
        else:
            print(f'Invalid plot type "{plot_type}"')
            sys.exit(1)

        axs[0][idx].grid(linestyle="--", linewidth=0.5)
        axs[1][idx].grid(linestyle="--", linewidth=0.5)

        color_test = np.asarray([get_color("test")] * len(indices))
        color_train = np.asarray([get_color("train")] * len(indices))
        # point = dat['train'][start:end].index == f'offset_{args.offset + 1}'
        # color_test[point] = 'g'
        # color_train[point] = 'g'

        if plot_type.split("_")[0] == "speedup":
            axs[0][idx].scatter(
                y=dat["train"][start:end].to_numpy(),
                x=indices,
                label="train",
                color=color_train,
            )
            axs[0][idx].plot(
                [-0.5, len(indices) - 0.5], [2, 2], linestyle="--"
            )  # mark cutoff
            axs[1][idx].scatter(
                y=dat["test"][start:end].to_numpy(),
                x=indices,
                label="test",
                color=color_test,
            )

            if plot_type.split("_")[-1] == "delta":
                line = [args.cutoff, args.cutoff]
            else:
                line = [1, 1]
            axs[1][idx].plot([-0.5, len(indices) - 0.5], line, linestyle="--")

        elif plot_type.split("_")[0] == "raw":
            axs[0][idx].errorbar(
                y=dat["train_aop"][start:end].to_numpy(),
                x=indices,
                linestyle="none",
                yerr=dat["train_aop_std"][start:end].to_numpy(),
                c=get_color("aop"),
            )

            axs[0][idx].scatter(
                y=dat["train_aop"][start:end].to_numpy(),
                x=indices,
                color=dat["color_aop"][start:end],
                label="train-aop",
            )

            axs[0][idx].errorbar(
                y=dat["train_base"][start:end].to_numpy(),
                x=indices,
                linestyle="none",
                yerr=dat["train_base_std"][start:end].to_numpy(),
                c=get_color("base"),
            )

            axs[0][idx].scatter(
                y=dat["train_base"][start:end].to_numpy(),
                x=indices,
                color=dat["color_base"][start:end],
                label="train-base",
            )

            axs[1][idx].errorbar(
                y=dat["test_aop"][start:end].to_numpy(),
                x=indices,
                linestyle="none",
                yerr=dat["test_aop_std"][start:end].to_numpy(),
                c=get_color("aop"),
            )

            axs[1][idx].scatter(
                y=dat["test_aop"][start:end].to_numpy(),
                x=indices,
                color=dat["color_aop"][start:end],
                label="test-aop",
            )

            axs[1][idx].errorbar(
                y=dat["test_base"][start:end].to_numpy(),
                x=indices,
                linestyle="none",
                yerr=dat["test_base_std"][start:end].to_numpy(),
                c=get_color("base"),
            )

            axs[1][idx].scatter(
                y=dat["test_base"][start:end].to_numpy(),
                x=indices,
                color=dat["color_base"][start:end],
                label="test-base",
            )

            if args.stat_strategy in ["static", "all"] and plot_type.split("_")[
                -1
            ] not in ["diffp", "diff"]:
                axs[1][idx].plot(
                    [-0.5, len(indices) - 0.5],
                    [args.cutoff, args.cutoff],
                    linestyle="--",
                )  # mark cutoff

        axs[0][idx].legend()
        axs[1][idx].legend()

        if len(axs[0][idx].xaxis.get_ticklabels()) > 10:
            for label in axs[0][idx].xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            for label in axs[1][idx].xaxis.get_ticklabels()[::2]:
                label.set_visible(False)

    plt.tight_layout()
    if not args.no_save:
        pdf.savefig(fig)

    if args.visualize:
        plt.show()

    if not args.no_save:
        pdf.close()

    plt.close()


def diffp_analysis(dat: pd.DataFrame, args, fname):
    """Difference-Plus analysis.
    Consider only points where the mean+/- standard deviation do not overlap by
    "cutoff" amount.
    """
    min_base = dat["test_base"] - dat["test_base_std"]
    max_aop = dat["test_aop"] + dat["test_aop_std"]
    matching = min_base - max_aop > args.cutoff
    if fname is not None:
        plot_data(
            dat[matching],
            count_exps(dat[matching]),
            "raw_diffp",
            args,
            y_labels["raw"],
            fname,
        )

    return matching


def diff_analysis(dat: pd.DataFrame, args, fname):
    """Consider only points where the means are more than "cutoff" apart.
    Will plot the raw data values
    """
    matching = dat["test_base"] - dat["test_aop"] > args.cutoff
    if fname is not None:
        plot_data(
            dat[matching],
            count_exps(dat[matching]),
            "raw_diff",
            args,
            y_labels["raw"],
            fname,
        )

    return matching


def delta_analysis(dat: pd.DataFrame, args, fname):
    """Consider only points where the means do not overlap by "cutoff" amount.
    Differs from diff_analysis because it plots the deltas (and not the raw
    values)
    """
    tmp_dat = dat.copy(deep=True)
    tmp_dat["test"] = dat["test_base"] - dat["test_aop"]
    if fname is not None:
        plot_data(
            tmp_dat,
            count_exps(tmp_dat),
            "speedup_delta",
            args,
            y_labels["delta"],
            fname,
        )
    return tmp_dat["test"] > args.cutoff


def static_analysis(dat: pd.DataFrame, args, fname):
    '''Consider only values below the "cutoff"'''
    matching = dat["test_aop"] <= args.cutoff
    if fname is not None:
        plot_data(
            dat[matching],
            count_exps(dat[matching]),
            "raw_static",
            args,
            y_labels["raw"],
            fname,
        )
    return matching


def stat_analysis(dat: pd.DataFrame, stat_type: str, fname: str, args):
    """Call this function to process a specific stat analysis technique."""
    return {
        "diffp": diffp_analysis,
        "diff": diff_analysis,
        "static": static_analysis,
        "delta": delta_analysis,
    }.get(stat_type, lambda: "Invalid")(dat, args, fname)


def filter_data(dat: pd.DataFrame, s_strategies, args, fname=None):
    """For all loaded data, filter by {stat strategy, speedup-only, and raw
    timing values only}

    Will generate plots for statistics analysis when called.

    returns:
        speedup data
        statistics data
        raw train and test values
    """
    if dat is None:
        dat = load_data(args)

    # get data filters for all stat strategies
    matching = dict()
    if args.stat_strategy == "all":
        for stat_t in s_strategies[:-2]:
            matching[stat_t] = stat_analysis(dat, stat_t, fname, args)
    elif args.stat_strategy != "none":
        matching[args.stat_strategy] = stat_analysis(
            dat, args.stat_strategy, fname, args
        )

    spd_dat = {
        "name": [],
        "mean": [],
        "std": [],
        "n_mean": [],
        "aop": [],
        "aop_std": [],
        "base": [],
        "base_std": [],
        "n_aop": [],
        "n_aop_std": [],
        "n_base": [],
        "n_base_std": [],
    }
    stats_dat = dict()
    r_dat = dict()
    for cfg_name in {"_".join(idx_name.split("_")[:2]): 0 for idx_name in dat.index}:
        idx_name = dat.index.str.match(f"{cfg_name}_")

        # filter out irrelevant dat -- keep only testing and training RAW
        # VALUES
        r_dat[cfg_name.split("_")[-1]] = dat[idx_name][
            ["test_aop", "test_base", "train_aop", "train_base"]
        ]
        r_dat[cfg_name.split("_")[-1]].index = r_dat[cfg_name.split("_")[-1]].index.map(
            lambda x: int(x.split("_")[-1])
        )

        # filter data according to stat strategy filters
        # has no effect if no stat strategies are being used
        for stat_t in matching:
            idx_match = dat[matching[stat_t]].index.str.match(f"{cfg_name}_")
            # match_names = data[matching[stat_t]][idx_match].index
            match_val = dat[matching[stat_t]][idx_match]["test_aop"].to_numpy()
            if args.boolean:
                match_val = [True] * len(match_val)
            if stats_dat.get(stat_t) is None:
                stats_dat[stat_t] = dict()
            stats_dat[stat_t][cfg_name.split("_")[-1]] = {
                int(x.split("_")[-1]): match_val[i]
                for i, x in enumerate(dat[matching[stat_t]][idx_match].index)
            }

            # if spd_dat.get(stat_t) is None:
            #     spd_dat[stat_t] = list()
            # spd_dat[stat_t].append([int(x.split('_')[-1]) for x in
            #                         match_names])

        # speedup data
        num_accesses = cfg_name.split("_")[-1]
        if idx_name.sum() == 0:
            print(f"{cfg_name} had {idx_name.sum()} entries!")
        spd_dat["name"].append(num_accesses)
        spd_dat["aop"].append(dat[idx_name]["train_aop"].mean())
        spd_dat["aop_std"].append(dat[idx_name]["train_aop_std"].std())
        spd_dat["base"].append(dat[idx_name]["train_base"].mean())
        spd_dat["base_std"].append(dat[idx_name]["train_base_std"].std())

        norm_val = args.num_accesses or int(num_accesses)
        spd_dat["n_aop"].append(
            (dat[idx_name]["train_aop"] * args.unit_conversion / norm_val).mean()
        )
        spd_dat["n_aop_std"].append(
            (dat[idx_name]["train_aop_std"] * args.unit_conversion / norm_val).std()
        )
        spd_dat["n_base"].append(
            (dat[idx_name]["train_base"] * args.unit_conversion / norm_val).mean()
        )
        spd_dat["n_base_std"].append(
            (dat[idx_name]["train_base_std"] * args.unit_conversion / norm_val).std()
        )

        # (dat[idx_name]['train'] / int(num_accesses)).mean())
        spdup = (dat[idx_name]["train_base"]) / (dat[idx_name]["train_aop"])
        spd_dat["mean"].append(spdup.mean())
        spd_dat["n_mean"].append(spdup.mean() - 1)
        spd_dat["std"].append(spdup.std())

    r_dat = pd.concat(r_dat, axis=1)
    r_dat.index = r_dat.index + args.test_shift

    return pd.DataFrame.from_records(spd_dat, index="name"), stats_dat, r_dat


def plot_stat_hmaps(s_dat, fname, args):
    sns.set(rc={"figure.figsize": (15, 8)})

    spd_dfs = dict()
    # individual heatmaps
    for exp in s_dat:
        spd_dfs[exp] = pd.DataFrame.from_records(s_dat[exp])
        _sort = sorted(spd_dfs[exp].columns, key=lambda x: int(x))
        spd_dfs[exp] = spd_dfs[exp].reindex(_sort, axis=1)
        spd_dfs[exp] = spd_dfs[exp].sort_index(ascending=True)
        spd_dfs[exp] = spd_dfs[exp].fillna(args.cutoff if not args.boolean else False)
        ax = sns.heatmap(spd_dfs[exp], cmap="Blues_r", cbar=not args.boolean)
        ax.set_title(f"{exp} heatmap")
        ax.set_ylabel(args.y_label or DEFAULT_YLABEL)
        ax.set_xlabel(args.x_label)
        plt.tight_layout()
        plt.savefig(f"{fname}_{exp}_heat.pdf")
        print(f"Saved {fname}_{exp}_heat.pdf")
        plt.close()

    # all stats on a single heatmap
    if len(spd_dfs) > 0:
        spd_df = pd.concat(spd_dfs, axis=1)
        spd_df = spd_df.fillna(args.cutoff if not args.boolean else False)
        ax = sns.heatmap(spd_df, cmap="Blues_r", cbar=not args.boolean)
        ax.set_title("Comparison of all strategies heatmap")
        ax.set_ylabel(args.y_label or DEFAULT_YLABEL)
        ax.set_xlabel(args.x_label)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{fname}_all_heat.pdf")
        print(f"Saved {fname}_all_heat.pdf")
        plt.close()


def get_line_locs(ticklabels, cl_breaks, idx, offset) -> list:
    # if len(cl_locs) == 0:
    #     return []
    # cl_breaks = iter(cl_locs)
    n_cl = next(cl_breaks)
    line_locs = list()
    prev_loc = None
    prev_label = -1
    for ticklbl in ticklabels:
        label = ticklbl.get_text()
        loc = ticklbl.get_position()[idx]

        try:
            if int(label) == n_cl:
                line_locs.append(loc + offset)
                # print(f'==> {n_cl} {line_locs[-1]}')
                n_cl = next(cl_breaks)

            elif int(prev_label) < n_cl < int(label):
                # line_locs.append((loc + prev_loc) / 2)
                dist = loc - prev_loc
                dist2 = int(label) - int(prev_label)
                dist3 = n_cl - int(prev_label) + offset
                line_locs.append(prev_loc + dist * (dist3 / dist2))
                # print(f'--> {n_cl} {line_locs[-1]}')
                n_cl = next(cl_breaks)
        except StopIteration:
            break

        # print(label, loc)
        prev_loc = loc
        prev_label = label

    # print(line_locs)
    return line_locs


def cm(x: int) -> float:
    """Convert from inches to cm"""
    _cm = 1 / 2.54
    return _cm * x


def get_ticks(current_ticks, desired_ticks, idx):
    """Figure out where the ticks should be based on the current ticks"""
    print(current_ticks)
    print(desired_ticks)
    graph_ticks = iter(current_ticks)
    c_tick = next(graph_ticks)
    tick_locs = []
    tick_lbls = []
    prev_loc = c_tick.get_position()[idx]
    prev_label = int(c_tick.get_text())
    loc = c_tick.get_position()[idx]
    label = int(c_tick.get_text())

    advance = False

    for _t in desired_ticks:
        print(prev_label, _t, label)
        if label == int(_t):
            print("eq")
            tick_locs.append(c_tick.get_position()[idx])
            tick_lbls.append(_t)
            advance = True

        elif label > int(_t):
            print("gt", loc, prev_loc)
            dist = loc - prev_loc
            dist2 = label - prev_label
            dist3 = int(_t) - prev_label
            tick_locs.append(prev_loc + dist * (dist3 / dist2))
            tick_lbls.append(_t)
            # advance = True

        if advance:
            prev_label = label
            prev_loc = loc

            try:
                c_tick = next(graph_ticks)
            except StopIteration as _e:
                print(tick_locs)
                print(
                    "\n".join(
                        [f"{lbl}: {loc}" for lbl, loc in zip(tick_lbls, tick_locs)]
                    )
                )
                raise _e

            loc = int(c_tick.get_position()[idx])
            label = int(c_tick.get_text())
            advance = False

    print(tick_locs)
    print("\n".join([f"{lbl}: {loc}" for lbl, loc in zip(tick_lbls, tick_locs)]))
    return None


BORDER = 0.90
BORDER_SINGLE = 0.775
TITLESIZE = 13
LABELSIZE = 9
YSIZE = 9
GUIDEWIDTH = 0.7


def plot_raw_hmaps(
    r_dat,
    fname,
    plttype,
    args,
    rc,
    layout=[0, 0, 0.9, 1],
    cbar_layout=[0.90, 0.3, 0.03, 0.4],
):
    """Plot the raw timing heatmaps"""
    sns.set(rc=rc)

    raw_aop = r_dat.xs(f"{plttype}_aop", axis=1, level=1)[args.heat_min : args.heat_max]
    raw_bas = r_dat.xs(f"{plttype}_base", axis=1, level=1)[
        args.heat_min : args.heat_max
    ]
    if args.zero_nan:
        raw_aop = raw_aop.fillna(0)
        raw_bas = raw_bas.fillna(0)

    if args.normalize:
        intkeys = raw_aop.keys().astype(int)
        entry_count = len(r_dat)
        if args.heat_max is not None and args.heat_max < len(r_dat):
            entry_count = args.heat_max
        h_min = args.heat_min or 0
        # assert(h_min < entry_count), 'Too few entries'
        entry_count -= h_min
        assert entry_count > 0, "Too few entries, invalid heat_min?"
        # assert(args.heat_min is None), 'Minimum not currently supported...'

        # keep a list of unique index values
        idx_names = {}
        for k in intkeys:
            for val in range(k, k + entry_count):
                idx_names[val] = 1

        idx_values = list(idx_names.keys())
        idx_values.sort()
        norm_aop = {"index": idx_values}
        norm_bas = {"index": idx_values}
        for k in raw_aop.keys():
            norm_aop[str(int(k) - 1)] = [None] * len(idx_values)
            norm_bas[str(int(k) - 1)] = [None] * len(idx_values)
            start_val = idx_values.index(int(k))
            # start_val = int(k) - intkeys[0]
            # print(k, start_val, start_val + entry_count)
            norm_aop[str(int(k) - 1)][start_val : start_val + entry_count] = raw_aop[k]
            norm_bas[str(int(k) - 1)][start_val : start_val + entry_count] = raw_bas[k]

        norm_aop_df = pd.DataFrame.from_records(norm_aop, index="index")
        norm_bas_df = pd.DataFrame.from_records(norm_bas, index="index")

        norm_aop_df = norm_aop_df.sort_index(axis=1, key=lambda col: col.map(int))
        norm_aop_df.index = norm_aop_df.index + args.test_shift + h_min - 1

        norm_bas_df = norm_bas_df.sort_index(axis=1, key=lambda col: col.map(int))
        norm_bas_df.index = norm_bas_df.index + args.test_shift + h_min - 1

    fig, axs = plt.subplots(1, 2 if not args.no_base else 1, sharex=True, sharey=True)

    # controls placement of colorbar
    cbar_ax = fig.add_axes(cbar_layout)
    if args.no_base:
        axs = [axs]

    if not args.paper:
        fig.suptitle(args.title or fname)

    ylabel = args.y_label or (
        DEFAULT_NORMHMAP_YLABEL if args.normalize else DEFAULT_YLABEL
    )

    if args.highlight:
        cmap = sns.diverging_palette(255, 10, as_cmap=True, sep=int(args.cutoff))
    else:
        # cmap = args.cmap or "Blues_r"
        cmap = args.cmap or sns.color_palette("hls", 7)

    hmapdf_aop = norm_aop_df if args.normalize else raw_aop
    hmapdf_bas = norm_bas_df if args.normalize else raw_bas
    plot = sns.heatmap(
        (hmapdf_aop - args.measurement_overhead) * args.unit_conversion,
        cmap=cmap,
        ax=axs[0],
        cbar=True,
        cbar_ax=cbar_ax,
    )

    if not args.no_base:
        axs[0].set_title(args.aop_label or f"AOP {plttype} times")
        # fontsize=TITLESIZE)
    axs[0].set_ylabel(ylabel)  # , fontsize=LABELSIZE)
    axs[0].set_xlabel(args.x_label)  # , fontsize=LABELSIZE)
    axs[0].invert_yaxis()

    cbar = axs[0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=YSIZE, pad=0)
    cbar.ax.locator_params(nbins=7)
    if args.no_base:
        pad = 2.5
    else:
        pad = 1.5
    cbar.set_label(label=f"Access Time ({args.units})", size=TITLESIZE, labelpad=pad)

    if args.normalize and args.guidelines:
        y_lines = get_line_locs(
            axs[0].get_yticklabels(),
            iter([int(v) for v in hmapdf_aop.index if int(v) % 16 == 0]),
            1,
            -0.5,
        )
        x_lines = get_line_locs(
            axs[0].get_xticklabels(),
            iter([int(v) for v in hmapdf_aop.columns if int(v) % 16 == 0]),
            0,
            -0.5,
        )

        axs[0].hlines(
            y_lines,
            *axs[0].get_xlim(),
            color="black",
            linestyle="--",
            linewidth=GUIDEWIDTH,
        )

        axs[0].vlines(
            x_lines,
            *axs[0].get_ylim(),
            color="black",
            linestyle="--",
            linewidth=GUIDEWIDTH,
        )
    elif args.guidelines:  # not normalizing, including guidelines
        y_lines = get_line_locs(
            axs[0].get_yticklabels(), itertools.count(16, 16), 1, 0.5
        )
        axs[0].hlines(
            y_lines,
            *axs[0].get_xlim(),
            color="black",
            linestyle="--",
            linewidth=GUIDEWIDTH,
        )

    if args.no_base:
        xlabels = [f"{x}" for x in hmapdf_aop.columns if int(x) % 2 == 0]
        plot.set_xticks(
            [v + 0.6 for v in range(1, len(hmapdf_aop.columns), 2)], minor=False
        )
        plot.set_xticklabels(xlabels)

    # always show all x axis labels
    plot.locator_params(axis="x", nbins=len(hmapdf_bas.columns))

    # in the case of args.no_base, these are set below
    if not args.normalize and not args.no_base:
        plot.set_yticks([v for v in range(0, 81, 8)], minor=False)
        plot.set_yticklabels([v for v in range(0, 81, 8)], minor=False, rotation=0)
    else:
        plot.set_yticks(
            [v + 0.25 for v in range(0, len(hmapdf_aop.index), 8)], minor=False
        )
        plot.set_yticklabels(
            [v for v in hmapdf_aop.index if (v - hmapdf_aop.index[0]) % 8 == 0],
            minor=False,
        )

    plot.xaxis.labelpad = 2
    plot.yaxis.labelpad = 1

    if not args.no_base:
        plot = sns.heatmap(
            hmapdf_bas - args.measurement_overhead,
            cmap=cmap,
            ax=axs[1],
            cbar=False,
            cbar_ax=None,
        )
        axs[1].set_title(args.base_label or f"Base {plttype} times")
        # fontsize=TITLESIZE)
        if not args.paper:
            axs[1].set_ylabel(ylabel)  # , fontsize=LABELSIZE)
        else:
            axs[1].set_ylabel("")
        axs[1].set_xlabel(args.x_label)  # , fontsize=LABELSIZE)
        axs[1].invert_yaxis()
        if args.guidelines:
            axs[1].hlines(
                y_lines,
                *axs[1].get_xlim(),
                color="black",
                linestyle="--",
                linewidth=GUIDEWIDTH,
            )
            if args.normalize:
                axs[1].vlines(
                    x_lines,
                    *axs[1].get_ylim(),
                    color="black",
                    linestyle="--",
                    linewidth=GUIDEWIDTH,
                )

        # always show all x axis labels
        plot.locator_params(axis="x", nbins=len(hmapdf_bas.columns))

    # y axis labels
    if not args.normalize:
        ticklbls = list(range(0, hmapdf_aop.index[-1] + 2, 8))
        # ticklbls = [v for v in range(0, args.heat_max + 1, 8)]
        ticklocs = [v - 0.5 for v in range(0, hmapdf_aop.index[-1] + 2, 8)]
        if args.heat_min == 1:
            ticklbls[0] = 1
            ticklocs[0] += 1
        plot.set_yticks(ticklocs, minor=False)
        plot.set_yticklabels(ticklbls, minor=False, rotation=0)

    plot.xaxis.labelpad = 2
    plot.yaxis.labelpad = 1

    # controls placement/sizing of heatmap
    if args.paper and not args.no_base:
        plt.subplots_adjust(wspace=0.025)

    plt.tight_layout(rect=layout)

    plt.savefig(f"{fname}_{plttype}_heat.pdf")
    print(f"Saved {fname}_{plttype}_heat.pdf")
    plt.close()


def plot_delta_hmaps(r_dat, fname, args):
    """Plots a heatmap of the difference between aop and baseline timings"""
    # sns.set(rc={'figure.figsize': (15, 8)})

    # cmap = sns.diverging_palette(240, 240, as_cmap=True)
    _ax = sns.heatmap(
        r_dat.xs("test_base", axis=1, level=1)[args.heat_min : args.heat_max]
        - r_dat.xs("test_aop", axis=1, level=1)[args.heat_min : args.heat_max],
        cmap="coolwarm",
        center=0.0,
    )
    _ax.set_title("Base - AOP Test times")
    _ax.set_ylabel(args.y_label or DEFAULT_YLABEL)
    _ax.set_xlabel(args.x_label)
    _ax.invert_yaxis()
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"{fname}_testdelta_heat.pdf")
    print(f"Saved {fname}_testdelta_heat.pdf")
    plt.close()


def setup_color(dat: pd.DataFrame, args):
    """Setup colors to use in non-heatmaps"""
    color_init = np.asarray([marker_colors["base"]] * len(dat))
    if args.highlight:
        color_init[dat["test_base"] < args.cutoff] = marker_colors["base_special"]
    dat["color_base"] = [(r / 255, g / 255, b / 255) for r, g, b in color_init]

    color_init = np.asarray([marker_colors["aop"]] * len(dat))
    if args.highlight:
        color_init[dat["test_aop"] < args.cutoff] = marker_colors["aop_special"]
    dat["color_aop"] = [(r / 255, g / 255, b / 255) for r, g, b in color_init]
    return dat


def add_args(parser=None):
    """Add visualization arguments to argparser"""
    if parser is None:
        parser = configargparse.ArgParser()

    plot_group = parser.add_argument_group("Options for all plots")
    stat_group = parser.add_argument_group("Statistics visualizers")
    heat_group = parser.add_argument_group("Heatmap options")

    parser.add_argument(
        "--debug", action="store_true", help="Enable more output messages"
    )
    parser.add_argument(
        "--no_sort", action="store_true", help="Do not sort loaded data"
    )

    parser.add("-c", "--config", is_config_file=True, help="Config file path")

    plot_group.add_argument("--x_label", default="Train Length", type=str)
    plot_group.add_argument("--y_label", default=None, type=str)
    plot_group.add_argument("--aop_label", default=None, type=str)
    plot_group.add_argument("--base_label", default=None, type=str)
    plot_group.add_argument(
        "--num_accesses",
        default=None,
        type=int,
        help="Explicitly state number of accesses for "
        "comparison normalization (instead of inferring)",
    )
    plot_group.add_argument(
        "--max_x", help="Max value in x axis", default=None, type=int
    )
    plot_group.add_argument(
        "--sku_plot",
        help="Plot SKU training comparison if a name is passed",
        default=None,
        type=str,
    )

    heat_group.add_argument(
        "--paper",
        action="store_true",
        help="Size and format figures for paper instead of for exploration",
    )
    heat_group.add_argument(
        "--no_base", action="store_true", help="Do not plot the baseline heatmap"
    )
    heat_group.add_argument("--units", default="UNITS", help="Heatmap units")
    heat_group.add_argument(
        "--cmap", default=None, type=str, help="Do not plot the baseline heatmap"
    )
    heat_group.add_argument(
        "--normalize",
        action="store_true",
        help="For the timing heatmap, normalize across "
        + "the y-axis so hmap entries match test index "
        + "and not distance from end of train",
    )
    heat_group.add_argument(
        "--zero_nan",
        action="store_true",
        help="Treat NaN entries as zeros in the graph",
    )
    heat_group.add_argument(
        "--measurement_overhead",
        default=0,
        type=float,
        help="Measurement overhead (will subtract from timing data)",
    )
    heat_group.add_argument(
        "--unit_conversion",
        default=1,
        type=float,
        help="Multiplier for unit conversion",
    )
    heat_group.add_argument("--title", default=None, type=str)
    heat_group.add_argument(
        "--omit", action="store_true", help="Omit outliers in heatmap"
    )

    heat_group.add_argument(
        "--guidelines",
        action="store_true",
        help="Show cacheline guides on timing heatmap",
    )
    heat_group.add_argument(
        "--visualize", action="store_true", help="Show plots instead of only saving"
    )

    heat_group.add_argument(
        "--train_heat",
        action="store_true",
        help="Use training times instead of test times in the timing heatmap",
    )

    heat_group.add_argument(
        "--heat_delta",
        action="store_true",
        help="Generate heatmap of timing deltas between baseline and aop",
    )
    heat_group.add_argument(
        "--heat_stats", action="store_true", help="Generate heatmaps for statistics"
    )

    heat_group.add_argument(
        "--heat_max",
        default=None,
        type=int,
        help="Maximum number of entries from end-of-train to show",
    )
    heat_group.add_argument(
        "--heat_min",
        default=None,
        type=int,
        help="Skip N number of entries after end-of-train",
    )
    heat_group.add_argument(
        "--test_shift",
        default=0,
        type=int,
        help="Shift y-axis of data. Useful for PIDexp2, "
        + "where P1 performs additional training accesses",
    )

    heat_group.add_argument(
        "--highlight",
        action="store_true",
        help="Color special points differently (non-heatmaps)",
    )
    plot_group.add_argument(
        "--cutoff",
        default=25.0,
        type=float,
        help="Cutoff value for statistics and heatmap highlighting",
    )

    stat_group.add_argument(
        "--boolean",
        action="store_true",
        help="For statistics, show which values which made the cutoff as booleans",
    )
    stat_group.add_argument(
        "--max_exps",
        type=int,
        default=10,
        help="Max number of graphs to show per (non-hmap) figure",
    )
    stat_group.add_argument(
        "--plot_all",
        action="store_true",
        help="Don't just generate the timing heatmap, generate all plots",
    )
    stat_group.add_argument("--no_save", action="store_true", help="Don't save plots.")

    plots = ["speedup", "raw", "all"]
    assert plots[-1] == "all", 'Add new plot types before "all"'
    stat_group.add_argument("--plot", default=plots[0], choices=plots)

    strategies = ["static", "delta", "diffp", "diff", "all", "none"]
    assert strategies[-1] == "none", 'Add new plot types before "all"'
    assert strategies[-2] == "all", 'Add new plot types before "all"'
    stat_group.add_argument(
        "--stat_strategy",
        default=strategies[-1],
        choices=strategies,
        help="Which strategy to use for statistics? "
        + " One of: "
        + "\n\t".join(strategies),
    )

    return parser, plots, strategies


if __name__ == "__main__":
    prsr, plot_types, stat_strategies = add_args()
    prsr.add_argument(
        "fname",
        help="Log file to process",
        metavar="FILE",
        type=lambda x: is_valid_file(prsr, x),
    )

    arguments, _ = prsr.parse_known_args()
    # print(arguments)

    all_dat = load_data(arguments)

    EXP_FNAME = "_".join(arguments.fname.split(".")[:-1])

    if arguments.plot == "all":
        for plot_t in plot_types[:-1]:
            plot_data(
                all_dat,
                count_exps(all_dat),
                plot_t,
                arguments,
                y_labels[plot_t],
                EXP_FNAME,
            )
    else:
        plot_data(
            all_dat,
            count_exps(all_dat),
            arguments.plot,
            arguments,
            y_labels[arguments.plot],
            EXP_FNAME,
        )

    spd_df, stat_dat, raw_dat = filter_data(all_dat, stat_strategies, arguments)

    comp_fname = f"{EXP_FNAME}_train_comparison.csv"
    spd_df.to_csv(comp_fname)
    print(f"Saved {comp_fname}")

    # SKU stuff
    if arguments.sku_plot is not None:
        sku_path = f"sku_raws_{arguments.sku_plot}"
        os.makedirs(sku_path, exist_ok=True)
        plt.errorbar(
            spd_df.index,
            spd_df["n_aop"],
            yerr=spd_df["n_aop_std"],
            label=arguments.sku_plot,
        )
        plt.errorbar(
            spd_df.index, spd_df["n_base"], yerr=spd_df["n_base_std"], label="Baseline"
        )
        plt.tight_layout()
        plt.legend()
        plt.title(EXP_FNAME)
        plt.savefig(f"{sku_path}/{EXP_FNAME}_train_raw.pdf")
        plt.close()

        plt.bar(x=spd_df.index, height=spd_df["mean"], yerr=spd_df["std"])
        plt.ylim([-2, 2])
        plt.tight_layout()
        plt.title(EXP_FNAME)
        plt.savefig(f"{sku_path}/{EXP_FNAME}_train_comparison.pdf")
        plt.close()

    # Plot the statistics heatmaps
    if arguments.stat_strategy != "none" and (
        arguments.plot_all or arguments.heat_stats
    ):
        plot_stat_hmaps(stat_dat, EXP_FNAME, arguments)

    # plot the raw data heatmaps
    # test time heatmap

    rc = {"figure.figsize": (15, 8)}
    layout = [0, 0, 0.9, 1]
    cbar_layout = [0.90, 0.3, 0.03, 0.4]

    if arguments.paper:
        rc.update(
            {
                "xtick.labelsize": LABELSIZE,
                "ytick.labelsize": LABELSIZE,
                "axes.titlesize": TITLESIZE,
                "xtick.major.pad": -2,
                "xtick.major.size": YSIZE - 2,
                "ytick.major.pad": -3,
                "ytick.major.size": YSIZE - 2,
            }
        )

        if (
            arguments.no_base
            and len(raw_dat.xs("train_aop", axis=1, level=1).columns) <= 5
        ):
            rc["figure.figsize"] = (cm(5), cm(10))
            layout = [-0.075, -0.045, BORDER_SINGLE + 0.075, 1.035]
            cbar_layout = [BORDER_SINGLE, 0.3, 0.03, 0.4]

        else:  # has baseline or many columns
            rc["figure.figsize"] = (cm(16), cm(11))
            layout = [-0.025, -0.039, BORDER + 0.020, 1.035]
            cbar_layout = [BORDER - 0.005, 0.3, 0.02, 0.4]
            # old configs:
            # cbar_layout = [BORDER - 0.03, .3, .03, .4]
            # layout = [-0.025, -0.045, BORDER + 0.000, 1.035]
            # rc['figure.figsize'] = (cm(13.75), cm(10))
            # rc['figure.figsize'] = (cm(11), cm(10))

    if arguments.omit:
        plot_raw_hmaps(
            raw_dat[raw_dat < 19], EXP_FNAME, "test", arguments, rc, layout, cbar_layout
        )
    else:
        plot_raw_hmaps(raw_dat, EXP_FNAME, "test", arguments, rc, layout, cbar_layout)

    if arguments.train_heat:  # also plot train
        plot_raw_hmaps(raw_dat, EXP_FNAME, "train", arguments, rc, layout, cbar_layout)

    # Delta heatmap
    if arguments.plot_all or arguments.heat_delta:
        plot_delta_hmaps(raw_dat, EXP_FNAME, arguments)

    print(f"Finished {EXP_FNAME}")
