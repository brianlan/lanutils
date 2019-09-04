import pathlib

import numpy as np


def get_indices(ind_path, root_dir: pathlib.Path, suffix, random_sample=None):
    """
    Create file indices list either from indices_path or go through root_dir.
    :param ind_path: if not None, will return indices from this file, otherwise by traversing root_dir.
    :param root_dir: file root directory
    :param suffix: file suffix of target files
    :param random_sample: (optional) if given, will random sample this number of data and return.
    :return: a list of path (str)
    """
    _sfx = suffix.strip(".")
    if ind_path:
        with open(ind_path, "r") as f:
            indices = [l.strip() for l in f.readlines()]
    else:
        indices = sorted([str(p.relative_to(root_dir).with_suffix("")) for p in root_dir.rglob(f"*.{_sfx}")])
    n_samples_to_draw = min(len(indices), random_sample or len(indices))
    return np.random.choice(indices, size=n_samples_to_draw).tolist()
