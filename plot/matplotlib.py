import matplotlib.pyplot as plt


def get_lefttop_coord_ax(axes_range=(0, 1, 0, 1)):
    plt.figure()
    ax = plt.gca()
    plt.axis(axes_range)
    ax.set_aspect(1)
    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
    ax.xaxis.tick_top()  # and move the X-Axis
    return ax


