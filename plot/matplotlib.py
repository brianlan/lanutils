import matplotlib.pyplot as plt
import numpy as np


def get_lefttop_coord_ax(axes_range=(0, 1, 0, 1)):
    plt.figure()
    ax = plt.gca()
    plt.axis(axes_range)
    ax.set_aspect(1)
    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
    ax.xaxis.tick_top()  # and move the X-Axis
    return ax


def get_colors(n_desired_colors, cmap_name="rainbow"):
    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap(cmap_name, n_desired_colors)
    try:
        colors = cmap.colors
    except AttributeError:
        colors = cmap(np.linspace(0, 1, n_desired_colors))
    colors = (colors[:, :3] * 255).astype(np.int32)
    return colors.tolist()
