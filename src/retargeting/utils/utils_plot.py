import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator

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


# 用于绘制axis比例相同的三维图
def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean

    x_mean = mean(xlim)
    y_mean = mean(ylim)
    z_mean = mean(zlim)

    plot_radius = max(
        [abs(lim - mean_) for lims, mean_ in ((xlim, x_mean), (ylim, y_mean), (zlim, z_mean)) for lim in lims]
    )

    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])


def plotHistogram(  # noqa: PLR0913
    data,
    x_labels,
    bar_labels,
    bar_colors,
    data_bottom=None,
    x_width=0.8,
    bar_interval_ratio=0.0,
    border_width=0.0,
    edgecolor="k",
    linewidth=1,
    use_default_setting=True,
    scatter_points=None,
    scatter_colors="k",
    scatter_marker="D",
    scatter_zorder=2,
):
    """
    Input:
        data: shape (n_x_axis, n_bars_per_x)
        scatter_points: shape (n_x_axis, n_bars_per_x, n_points_per_bar)
    """
    data = np.array(data)
    if data.shape[0] != len(x_labels) or data.shape[1] != len(bar_labels):
        print("The dimension of input data is wrong.")
        return False

    if data_bottom is None:
        data_bottom = np.zeros(data.shape)
    elif data_bottom.shape != data.shape:
        print("The dimension of input data_bottom is wrong.")
        return False

    x = np.arange(len(x_labels))
    m = len(x_labels)
    n = len(bar_labels)
    if n == 1:
        bar_interval = 0.0
    else:
        bar_interval = (x_width * bar_interval_ratio) / float(n - 1)
    bar_width = (x_width - (n - 1) * bar_interval) / n

    plt.bar(
        x + (n - 1) / 2.0 * (bar_width + bar_interval), np.zeros((data.shape[0],)), width=bar_width, tick_label=x_labels
    )
    for i in range(n):
        # plt.bar(
        #     x + i * (bar_width + bar_interval),
        #     data[:, i],
        #     bottom=data_bottom[:, i],
        #     width=bar_width,
        #     label=bar_labels[i],
        #     color=bar_colors[i],
        #     edgecolor=edgecolor,
        #     linewidth=linewidth,
        # )
        for i in range(n):
            if data.shape[1] == 1 and len(bar_colors) == data.shape[0]:
                color = bar_colors
            else:
                color = bar_colors[i] 

            hatch = None
            if n == 2 and i == 1: 
                hatch = "///"

            label = bar_labels[i] if i == 0 else None if bar_labels is not None else None
            plt.bar(
                x + i * (bar_width + bar_interval),
                data[:, i],
                bottom=data_bottom[:, i],
                width=bar_width,
                label=label,
                color=color,
                hatch=hatch,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )

        # scatter data points on the bar
        if scatter_points is not None:
            assert scatter_points.shape[0] == m
            assert scatter_points.shape[1] == n
            n_points_per_bar = scatter_points.shape[-1]
            for j in range(n_points_per_bar):
                plt.scatter(
                    x + i * (bar_width + bar_interval),
                    scatter_points[:, i, j],
                    color=scatter_colors[i],
                    zorder=scatter_zorder,
                    marker=scatter_marker,
                    edgecolors="k",
                )

    if use_default_setting:
        plt.xlim(
            [
                0 - bar_width / 2.0 - border_width,
                (m - 1) + (n - 1) * (bar_width + bar_interval) + bar_width / 2.0 + border_width,
            ]
        )


def changeTickerDensity(xaxis=None, yaxis=None):
    ax = plt.gca()
    if xaxis is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xaxis))
    if yaxis is not None:
        ax.yaxis.set_major_locator(MultipleLocator(yaxis))
