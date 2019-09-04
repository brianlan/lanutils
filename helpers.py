import numpy as np


def get_indices(root_dir, suffix, indices_path=None, random_sample=None):
    """
    Create file indices list either from indices_path or go through root_dir.
    :param root_dir: file root directory
    :param suffix: file suffix of target files
    :param indices_path: (optional) if given, will return indices from this file, otherwise by traversing root_dir.
    :param random_sample: (optional) if given, will random sample this number of data and return.
    :return: a list of path (str)
    """
    _sfx = suffix.strip(".")
    if indices_path:
        with open(indices_path, "r") as f:
            indices = [l.strip() for l in f.readlines()]
    else:
        indices = sorted([str(p.relative_to(root_dir).with_suffix("")) for p in root_dir.rglob(f"*.{_sfx}")])
    n_samples = min(len(indices), random_sample or len(indices))
    return np.random.choice(indices, size=n_samples).tolist()
