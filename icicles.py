#!/usr/bin/env python3
# Time-stamp: <2017-07-11 18:06:58 dangom>
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
if platform == "linux":
    import matplotlib
    matplotlib.use("agg")

import matplotlib.pyplot as plt # isort:skip
from matplotlib.patches import Rectangle # isort:skip


class Icicles():

    def __init__(self, icastruct, cleaning):
        """Init only needs an ICA structure with the information
        required to generate the Icicle plot.

        :param icastruct: PD Dataframe with [TotalVar/VarExp/Accept]
        :returns: None
        :rtype: None

        """
        self.icastruct = icastruct
        self.cleaning = cleaning

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
        return cls(data, "FIX")

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

        return cls(data, "ME-ICA")

    def icicle_plot(self, outfile):
        """
        Generate an icicle plot showing excluded and accepted components,
        and their explained variance.
        """
        data = self.icastruct
        ica_dimension = self.ncomponents

        fig, ax = plt.subplots()
        self.beautify_plot(ax)
        plt.axis([0, 100, 0.5, 1])

        rect_properties = {'edgecolor': "white", 'linewidth': 0.15}

        # Rectangle bottom left position, and its size.
        xy = (0, 0.5)
        width = float(data["ExplainedVariance"].loc[0])
        height = 0.5

        #print(xy, width)
        ax.add_patch(Rectangle(xy, width, height, fc="C0", **rect_properties))

        xpos = data["ExplainedVariance"].cumsum()
        for component in range(1, ica_dimension):
            xy = (float(xpos[component-1]), 0.5)
            width = float(data["ExplainedVariance"].loc[component])
            fc = "C0"
            #print(xy, width)
            ax.add_patch(Rectangle(xy, width, height, fc=fc, **rect_properties))
        xy = (self.explainedvariance, 0.5)
        width = 100 - self.explainedvariance
        #print(xy, width)
        ax.add_patch(Rectangle(xy, width, height, fc="C1", **rect_properties))

        # Quick check all components where accepted
        if data[data["Acceptance"]].shape[0] == data.shape[0]:
            var = self.explainedvariance
            plt.suptitle(f"Estimated ICA dimensionality retains {var:2.2f} % of variance")
            plt.savefig(outfile)
            return

        rejected_components = self.rejectedcomponents

        def get_component_colors():
            """Return a list of the same size as ICA_DIMENSION, where
            each item is either coloured "C1" if rejected, of "C0" if accepted.

            :param rejected_components: List of rejected components
            :param ica_dimension: Total ICA dimensionality
            :returns: List of colors
            :rtype: List

            """
            return ["C1" if item in set(rejected_components) else "C0"
                    for item in range(self.ncomponents)]

        colors = get_component_colors()

        plt.axis([0, 100, 0, 1])

        xy = (0, 0)
        width = float(data["ExplainedVariance"].loc[0])
        ax.add_patch(Rectangle(xy, width, height, fc=colors[0],
                               **rect_properties))

        for component in range(1, ica_dimension):
            xy = (float(xpos[component-1]), 0)
            width = float(data["ExplainedVariance"].loc[component])
            color = colors[component]
            ax.add_patch(Rectangle(xy,
                                   width,
                                   height,
                                   fc=color, **rect_properties))

        var = self.retainedvariance
        plt.suptitle(f"{self.cleaning} cleaned data retains {var:2.2f} % of variance")
        plt.savefig(outfile)

    def beautify_plot(self, ax):
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



    @property
    def ncomponents(self):
        return self.icastruct.shape[0]

    @property
    def rejectedcomponents(self):
        return self.icastruct[self.icastruct["Acceptance"] == False].index.tolist()

    @property
    def explainedvariance(self):
        return self.icastruct['ExplainedVariance'].sum()

    @property
    def retainedvariance(self):
        return self.icastruct[self.icastruct['Acceptance']]['ExplainedVariance'].sum()



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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Save an icicle plot to file")
    parser.add_argument('input_directory', metavar='i', type=str, default='.',
                        help='The input melodic directory')

    parser.add_argument('output_file', metavar='o', type=str,
                        help='The output filename')

    parser.add_argument('--fix', type=str, default=None,
                        help='If FIX, the txt containing fix classification results')

    args = parser.parse_args()

    # This fragile heuristic makes it so that the user does not have to tell us
    # which cleaning type he's refering to.
    if glob.glob(args.inputdirectory + "comp_table"):
        x = Icicles.frommeica(args.inputdirectory)
        x.icicle_plot(args.output_file)

    elif glob.glob(args.inputdirectory + "filtered_func_data"):
        x = Icicles.fromfsl(args.inputdirectory, fixfile=args.fixfile)
        x.icicle_plot(args.output_file)
