#!/usr/bin/env python3
# Time-stamp: <2017-07-11 17:14:53 dangom>
"""
Generate an icicle tree plot from a melodic directory.
The plot will explain how much variance was removed from cleaning the data,
and or changing the number of components.
"""

import ast
import glob
import os
import re
from sys import platform

import pandas as pd

# To circumvent issues when sshing into a  machine. Some backends
# will not be able to generate figures if no visual display is set.
# Fortunately, the agg backend can do so without a problem.
if "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use("agg")

import matplotlib.pyplot as plt # isort:skip
from matplotlib.patches import Rectangle # isort:skip


class Icicles():

    def __init__(self, icastruct):
        """Init only needs an ICA structure with the information
        required to generate the Icicle plot.

        :param icastruct: PD Dataframe with [TotalVar/VarExp/Accept]
        :returns: None
        :rtype: None

        """
        self.icastruct = icastruct

    @classmethod
    def fromfsl(cls, inputdirectory, fixfile=None):
        """Retrieves the data from FSL containing the following information:
        1. The component list
        2. Whether the component was accepted or rejected.
        3. The variance of the component within the ICA explained var.

        :param inputdirectory: Input Directory.
        :returns: DF with (at least) columns: [TotalVariance/VarianceExplained/Acceptance]
        :rtype: pandas DataFrame

        """
        cols = ["TotalVariance", "ExplainedVariance"]
        data = pd.read_csv(os.path.join(inputdirectory,
                                        "filtered_func_data.ica",
                                        "melodic_ICstats"),
                           delimiter="  ", header=None,
                           usecols=[0, 1], names=cols,
                           engine="python")

        if fixfile is None:  # In which case all components are accepted.
            rejected_components = []
        else:
            with open(fixfile, 'r') as f:
                for line in f:
                    x = line
                    # Ast.literal_eval safe evaluates a list.
                    # Subtract one because the counting starts at 1.
                rejected_components = [item - 1
                                       for item in ast.literal_eval(x)]

        data = data.assign(Acceptance=[x not in rejected_components
                                       for x in range(data.shape[0])])
        return cls(data)

    @classmethod
    def frommeica(cls, inputdirectory):
        """Retrieves the data from MEICA containing the following information:
        1. The component list
        2. Whether the component was accepted or rejected.
        3. The variance of the component within the ICA explained var.

        :param inputdirectory: Input Directory.
        :returns: DF with (at least) columns: [TotalVariance/VarianceExplained/Acceptance]
        :rtype: pandas DataFrame

        """
        cols = ["Kappa", "Rho", "TotalVariance"]
        compfile = os.path.join(inputdirectory, "comp_table.txt")
        data = pd.read_csv(compfile, delimiter="\t", comment="#",
                           header=None, usecols=[1, 2, 4], names=cols)

        def search_for(regexp):
            with open(compfile, 'r') as f:
                for line in f:
                    res = re.search(regexp, line)
                    if res is not None:
                        break
                return res.group(1)


        # Search for variance explained for ICA and for rejected components
        varexp_regexp = re.compile("\(VEx\): (\d*[.,]?\d*)")
        rejected_regexp = re.compile("REJ ([\d,]+)")

        varexplainedica = float(search_for(varexp_regexp))
        rejected_components = ast.literal_eval("[" + search_for(rejected_regexp) + "]")

        data = data.assign(Acceptance=[x not in rejected_components
                                       for x in range(data.shape[0])])

        data = data.assign(ExplainedVariance=[x*(varexplainedica/100)
                                              for x in data["TotalVariance"]])

        return cls(data)


    @property
    def ncomponents(self):
        return self.icastruct.shape[0]

    @property
    def explainedvariance(self):
        return self.icastruct['ExplainedVariance'].sum()

    @property
    def retainedvariance(self):
        return self.icastruct[self.icastruct['Acceptance']]['ExplainedVariance'].sum()


def retrieve_data_deprecated(melodic_dir):
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


def retrieve_data(melodic_dir):
    data = pd.read_csv(os.path.join(melodic_dir,
                                    "filtered_func_data.ica",
                                    "melodic_ICstats"),
                       delimiter="  ", header=None, engine="python")
    return data


def get_ica_dimension(melodic_dir):

    return sum(1 for line in open(os.path.join(melodic_dir,
                                               "filtered_func_data.ica",
                                               "melodic_ICstats")))


def get_variance_explained(data):
    return sum(data[1])


def get_fix_file(melodic_dir=None):
    """Find the fix4melview file, if available, in the
    Melodic output directory.

    :param melodic_dir: Path to directory
    :returns: Name of file
    :rtype: String

    """
    if melodic_dir is None:
        melodic_dir = os.getcwd()
    fixfile = glob.glob(os.path.join(melodic_dir, "fix4*.txt"))
    try:
        return fixfile[0]
    except IndexError:
        return None


def get_rejected_components_from_fix(fixfile):
    """The components are listed in the last line.
    We loop until the end of the file and then safe
    evaluate the list to retrieve its values.

    :param fixfile: The file as ouput from Melview (or similar).
    :returns: A list of rejected components
    :rtype: List

    """
    with open(fixfile, 'r') as f:
        for line in f:
            x = line
        # Ast.literal_eval safe evaluates the list.
        # Subtract one because the counting starts at 1.
        rejected_components = [item - 1 for item in ast.literal_eval(x)]
    return rejected_components


def get_variance_retained_after_fix(data, rejected_components):
    # var_per_component = data[0][1:].values - data[0][:-1].values
    # # Because we've skipped the first entry.
    # var_per_component = [float(data.loc[0])] + var_per_component.tolist()
    total_var = [x for i, x in enumerate(data[1])
                 if i not in set(rejected_components)]
    return float(sum(total_var))



def get_component_colors(rejected_components, ica_dimension):
    """Return a list of the same size as ICA_DIMENSION, where
    each item is either coloured "C1" if rejected, of "C0" if accepted.

    :param rejected_components: List of rejected components
    :param ica_dimension: Total ICA dimensionality
    :returns: List of colors
    :rtype: List

    """
    return ["C1" if item in set(rejected_components) else "C0"
            for item in range(ica_dimension)]


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
    #total_dimension = data.shape[0]

    fig, ax = plt.subplots()
    beautify_plot(ax)
    plt.axis([0, 100, 0.5, 1])

    rect_properties = {'edgecolor': "white", 'linewidth': 0.15}

    # Rectangle bottom left position, and its size.
    xy = (0, 0.5)
    width = float(data[1].loc[0])
    height = 0.5

    #print(xy, width)
    ax.add_patch(Rectangle(xy, width, height, fc="C0", **rect_properties))

    xpos = data[1].cumsum()
    for component in range(1, ica_dimension):
        xy = (float(xpos[component-1]), 0.5)
        width = float(data[1].loc[component])
        fc = "C0"
        #print(xy, width)
        ax.add_patch(Rectangle(xy, width, height, fc=fc, **rect_properties))
    xy = (get_variance_explained(data), 0.5)
    width = 100 - get_variance_explained(data)
    #print(xy, width)
    ax.add_patch(Rectangle(xy, width, height, fc="C1", **rect_properties))

    if fixfile is None:
        var = get_variance_explained(data)
        plt.suptitle(f"Estimated ICA dimensionality retains {var:2.2f} % of variance")
        plt.savefig(outfile)
        return

    fixfile = get_fix_file(melodic_dir)
    fix_rejected = get_rejected_components_from_fix(fixfile)
    fix_colors = get_component_colors(fix_rejected, ica_dimension)

    plt.axis([0, 100, 0, 1])

    xy = (0, 0)
    width = float(data[1].loc[0])
    ax.add_patch(Rectangle(xy, width, height, fc=fix_colors[0],
                           **rect_properties))

    for component in range(1, ica_dimension):
        xy = (float(xpos[component-1]), 0)
        width = float(data[1].loc[component])
        color = fix_colors[component]
        ax.add_patch(Rectangle(xy,
                               width,
                               height,
                               fc=color, **rect_properties))

    var = get_variance_retained_after_fix(data, fix_rejected)
    plt.suptitle(f"FIX cleaned data retains {var:2.2f} % of variance")
    plt.savefig(outfile)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Save an icicle plot to file")
    parser.add_argument('input_directory', metavar='i', type=str, default='.',
                        help='The input melodic directory')

    parser.add_argument('output_file', metavar='o', type=str,
                        help='The output filename')

    parser.add_argument('--fix', type=str, default=None,
                        help='The filename containing fix classification data')

    args = parser.parse_args()

    data = retrieve_data(args.input_directory)
    ica_dimension = get_ica_dimension(args.input_directory)
    icicle_plot(args.input_directory, args.output_file, fixfile=args.fix)
