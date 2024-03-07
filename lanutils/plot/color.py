import itertools
from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

# class CyclicColorGenerator:
#     def __init__(self, colormap_name="Set1") -> None:
#         self.colormap = plt.get_cmap(colormap_name)
#         self.num_colors = len(self.colormap.colors)
#         self.color_index = 0
    
#     def get_color(self):
#         color = self.colormap.colors[self.color_index][:3]
#         color = tuple(int(c * 255) for c in color)
#         hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
#         self.color_index = (self.color_index + 1) % self.num_colors
#         return hex_color


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return '#' + ''.join(f'{c:02x}' for c in rgb)

def get_colors(cmap_name: str = "rainbow", n_desired_colors: int = 5, representation: str = "rgb") -> List[Union[Tuple[int, int, int], str]]:
    cmap = plt.cm.get_cmap(cmap_name, n_desired_colors)
    try:
        colors = cmap.colors
    except AttributeError:
        colors = cmap(np.linspace(0, 1, n_desired_colors))
    colors = (colors[:, :3] * 255).astype(np.int32)
    if representation == "hex":
        colors = [rgb_to_hex(tuple(color)) for color in colors]
    elif representation == "rgb":
        colors = colors.tolist()
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    return colors

def color_generator(cmap_name: str = "rainbow", n_desired_colors: int = 5, cyclic: bool = True, representation: str = "rgb"):
    colors = get_colors(cmap_name=cmap_name, n_desired_colors=n_desired_colors, representation=representation)
    if cyclic:
        colors = itertools.cycle(colors)
    for color in colors:
        yield color
