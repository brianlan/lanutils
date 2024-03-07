from lanutils.plot.matplotlib import get_colors, color_generator


def test_get_colors():
    colors = get_colors(cmap_name="rainbow", n_desired_colors=3)
    