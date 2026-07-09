import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches

from utils.utils_plot import plotHistogram

params = {
    "font.family": "Times New Roman",
    #                     # 'font.style':'italic',
    #                     'font.weight':'normal', #or 'bold'
    "mathtext.fontset": "stix",
    "font.size": 24,  # or large,small
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
rcParams.update(params)

# pilot1 = [1, 0.733, 0.733, 0.733, 0, 0.067, 0.933, 0.8]
pilot1 = [1, 0, 0.067, 0.733, 0.733, 0.733, 0, 0.933]
# pilot2 = [1, 0.933, 0.667, 0.733, 0, 0.333, 0.867, 0.8]
pilot2 = [1, 0, 0.333, 0.667, 0.733, 0.933, 0, 0.867]

data = np.array([pilot1, pilot2]).T 

if __name__ == "__main__":

    save_dir = "outputs/success_rate/plots"
    os.makedirs(save_dir, exist_ok=True)

    group_colors = get_cmap("Pastel1")
    bar_colors = [[
        group_colors(0),  # Full
        group_colors(1), group_colors(1),  # A1, A2
        group_colors(2), group_colors(2),  # A3, A4
        group_colors(3), group_colors(3),  # A5, A6
        group_colors(4), group_colors(4),  # A7, A8
    ],
    [
        group_colors(0),  # Full
        group_colors(1), group_colors(1),  # A1, A2
        group_colors(2), group_colors(2),  # A3, A4
        group_colors(3), group_colors(3),  # A5, A6
        group_colors(4), group_colors(4),  # A7, A8
    ]]

    plt.figure(figsize=(5,7))
    plotHistogram(
        data=data,
        x_labels=["Full", "A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        bar_labels=["Pilot 1", "Pilot 2"],
        bar_colors=bar_colors,
        border_width=0.2,
    )

    plt.grid(axis="y")

    title_fontsize = 38
    plt.xlabel("Ablations", fontsize=title_fontsize)
    # plt.ylabel("Success Rate", fontsize=title_fontsize)
    plt.title("Task 3:\nSuccess Rate↑", fontsize=title_fontsize)
    # plt.xticks(rotation=45)

    patch1 = mpatches.Patch(facecolor='lightgray', label='Pilot 1', edgecolor='k')
    patch2 = mpatches.Patch(facecolor='lightgray', hatch='///', label='Pilot 2', edgecolor='k')

    # plt.legend(handles=[patch1, patch2], fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "task3_sr.png"), dpi=600)

    plt.show()
