"""
Generate an icicle tree plot from a melodic directory.
The plot will explain how much variance was removed from cleaning the data,
and or changing the number of components.
"""

import ast
import glob
import os

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def retrieve_data(melodic_dir):
    """Returns the cumulative sum of eigenvalues,
    as found in the 'eigenvalues_percent' file within the Melodic folder.

    :param melodic_dir: Path to melodic directory
    :returns: a pandas DataFrame with 1 value per row
    :rtype: pandas DataFrame

    """
    data = pd.read_csv(os.path.join(melodic_dir,
                                    "filtered_func_data.ica",
                                    "eigenvalues_percent"),
                       delimiter="  ", header=None, engine="python")
    return data.T


def get_ica_dimension(melodic_dir):

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


def get_rejected_components_from_fix(fixfile):
    with open(fixfile, 'r') as f:
        for line in f:
            # The components are listed in the last line.
            # Therefore we loop until the end of the file.
            x = line
        # Ast.literal_eval safe evaluates the list.
        # Subtract one because the counting starts at 1.
        rejected_components = [item - 1 for item in ast.literal_eval(x)]
    return rejected_components


def get_component_colors(rejected_components, ica_dimension):
    pass


def beautify_plot(ax):
    """Despine and remove ticks and ticklabels from axes object.

    :param ax: An Axes object
    :returns: Nothing
    :rtype: None

    """

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    return


def icicle_plot(melodic_dir, outfile, *, fixfile=None):
    """
    Generate an icicle plot showing excluded and accepted components, and their
    explained variance.
    """
    data = retrieve_data(melodic_dir)
    ica_dimension = get_ica_dimension(melodic_dir)
    total_dimension = data.shape[0]
    var = get_variance_explained(data, ica_dimension)

    fig, ax = plt.subplots()
    beautify_plot(ax)

    rect_properties = {'edgecolor': "white", 'linewidth': 0.2}

    xy = (0, 0.5)
    width = float(data.loc[0])
    height = 0.5

    ax.add_patch(Rectangle(xy, width, height, fc="C0", **rect_properties))

    for component in range(1, total_dimension):
        xy = (float(data.loc[component-1]), 0.5)
        width = float(data.loc[component] - data.loc[component-1])
        fc = "C0" if component < ica_dimension else "C1"
        ax.add_patch(Rectangle(xy, width, height, fc=fc, **rect_properties))

    if not fixfile:
        #plt.suptitle("Estimated " + ica_dimension + ". Retained " + 100*var "% of variance.")
        plt.savefig(outfile)
        return

    fixfile = get_fix_file(melodic_dir)
    test_fix = get_rejected_components_from_fix(fixfile)

    test_fix = ["C1" if item in set(test_fix) else "C0" for item in range(ica_dimension)]
    initcolor = "C1" if 0 in set(test_fix) else "C0"
    variance = data.loc[0] if 0 in set(test_fix) else 0
    ax.add_patch(Rectangle((0, 0), float(data.loc[0]), 0.5, fc=initcolor, edgecolor="white", linewidth=0.2))
    for component in range(1, ica_dimension):
        color = test_fix[component]
        size = float(data.loc[component] -data.loc[component-1])
        if color == "C0":
            variance += size
        ax.add_patch(Rectangle((float(data.loc[component-1]), 0),
                                       size,
                                       0.5,
                                       # alpha=1-float(component/total_dimension),
                                       fc=color, edgecolor="white", linewidth=0.2))

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
    ica_dimension = get_ica_dimension(args.input_directory)
    var = get_variance_explained(data, ica_dimension)
    icicle_plot(args.input_directory, args.output_file, fixfile=args.fix)
