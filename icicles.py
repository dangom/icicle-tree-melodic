"""
Generate an icicle tree plot from a melodic directory.
The plot will explain how much variance was removed from cleaning the data,
and or changing the number of components.
"""

import ast
import glob
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def retrieve_data(melodic_dir):
    data = pd.read_csv(os.path.join(melodic_dir,
                                    "filtered_func_data.ica",
                                    "eigenvalues_percent"),
                       delimiter="  ", header=None, engine="python")
    return data.T


def get_number_of_components(melodic_dir):
    return sum(1 for line in open(os.path.join(melodic_dir,
                                               "filtered_func_data.ica",
                                               "melodic_ICstats")))


def get_variance_explained(data, n_components):
    return float(data.loc[n_components])


def get_fix_file(melodic_dir=None):
    if melodic_dir is None:
        melodic_dir = os.getcwd()
    fixfile = glob.glob(os.path.join(melodic_dir, "fix4*.txt"))
    return fixfile[0]


def icicle_plot(melodic_dir, outfile, *, fixfile=None):
    """
    Generate an icicle plot showing excluded and accepted components, and their
    explained variance.
    """
    data = retrieve_data(melodic_dir)
    n_components = get_number_of_components(melodic_dir)
    var = get_variance_explained(data, n_components)


    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    # ax.add_patch(patches.Rectangle((0,0), explained, 1, facecolor="blue"))
    # ax.add_patch(patches.Rectangle((explained,0), 1- explained, 1, facecolor="red"))
    ax.add_patch(patches.Rectangle((0, 0.5), float(data.loc[0]), 1, facecolor="C0", edgecolor="white", linewidth=0.2))
    for component in range(1, n_components):
        size = float(data.loc[component] -data.loc[component-1])
        ax.add_patch(patches.Rectangle((float(data.loc[component-1]), 0.5),
                                       size,
                                       0.5,
                                       #alpha=1-float(component/data.shape[0]),
                                       facecolor="C0",edgecolor="white",linewidth=0.2))
    for component in range(n_components, data.shape[0]):
        size = float(data.loc[component] -data.loc[component-1])
        ax.add_patch(patches.Rectangle((float(data.loc[component-1]), 0.5),
                                       size,
                                       0.5,
                                       #alpha=1-float(component/data.shape[0]),
                                       facecolor="C1", edgecolor="white", linewidth=0.2))

    if not fixfile:
        plt.suptitle("Estimated " + n_components + ". Retained " + 100*var "% of variance.")
        plt.savefig(outfile)
        return

    fix = get_fix_file(melodic_dir)
    with open(fix, 'r') as f:
        for line in f:
            x = line
        test_fix = [item -1 for item in ast.literal_eval(x)]

    test_fix = ["C1" if item in set(test_fix) else "C0" for item in range(n_components)]
    initcolor = "C1" if 0 in set(test_fix) else "C0"
    variance = data.loc[0] if 0 in set(test_fix) else 0
    ax.add_patch(patches.Rectangle((0, 0), float(data.loc[0]), 0.5, facecolor=initcolor, edgecolor="white", linewidth=0.2))
    for component in range(1, n_components):
        color = test_fix[component]
        size = float(data.loc[component] -data.loc[component-1])
        if color = "C0":
            variance += size
        ax.add_patch(patches.Rectangle((float(data.loc[component-1]), 0),
                                       size,
                                       0.5,
                                       # alpha=1-float(component/data.shape[0]),
                                       facecolor=color, edgecolor="white", linewidth=0.2))

    plt.suptitle("")
    plt.savefig(outfile)



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Save an icicle plot to file")
    parser.add_argument('input_directory', metavar='i', type=str, default='.', help='The input melodic directory')
    parser.add_argument('output_file', metavar='o', type=str, help='The output filename')
    parser.add_argument('--fix', type=str, help='The filename containing fix classification data', default=None)

    args = parser.parse_args()

    data = retrieve_data(args.input_directory)
    n_components = get_number_of_components(args.input_directory)
    var = get_variance_explained(data, n_components)
    icicle_plot(args.input_directory, args.output_file, fixfile=args.fix)
