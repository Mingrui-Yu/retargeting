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

base_dir = "data/experiments/task2"

mct_1 = [] # mean completion time
mct_2 = []

for p in range(2):
    for i in range(8):
        if i == 0:
            file_name = f"pilot{p+1}/Full"
        else:
            file_name = f"pilot{p+1}/A{i}"
        # print(f"ablation_{i}")
        ablation_path = os.path.join(base_dir, file_name)

        folder = os.listdir(ablation_path)

        length = 0

        for folder_name in folder:
            folder_path = os.path.join(ablation_path, folder_name)
            data_path = os.path.join(folder_path, "data.npz")
            data = np.load(data_path)
            qpos = data["retarget_qpos"]
            length += len(qpos)

        average = length/len(folder)
        average *= 0.05
        if p == 0:
            mct_1.append(average)
        else:
            mct_2.append(average)

data = np.array([mct_1, mct_2]).T

if __name__ == "__main__":

    save_dir = "outputs/completion_time/plots"
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

    plt.figure(figsize=(5, 7))
    plotHistogram(
        data=data,
        x_labels=["Full", "A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        bar_labels=["Pilot 1", "Pilot 2"],
        bar_colors=bar_colors,
        border_width=0.2,
    )

    # # 设置纵轴范围和刻度
    # plt.ylim(0, 20)  # 设置纵轴范围为 0 到 20
    # plt.yticks([0, 4, 8, 12, 16, 20]) 

    title_fontsize = 38
    plt.xlabel("Ablations", fontsize=title_fontsize)
    # plt.ylabel("Mean Completion Time (s)", fontsize=title_fontsize)
    plt.title("Task 2:\nTask Time(s)↓", fontsize=title_fontsize)
    plt.grid(axis="y")

    
    patch1 = mpatches.Patch(facecolor='lightgray', label='Pilot 1', edgecolor='k')
    patch2 = mpatches.Patch(facecolor='lightgray', hatch='///', label='Pilot 2', edgecolor='k')

    # plt.legend(handles=[patch1, patch2], fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "task2_ct.png"), dpi=600)

    plt.show()
