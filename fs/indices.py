import pathlib
from collections import UserList

import numpy as np


class Indices(UserList):
    @classmethod
    def from_file(cls, path, common_prefix=None):
        """
        Create indices object from text file. Assumes only single column containing a path-like string in each line.

        :param path: file path
        :param common_prefix: Common path prefix shared by all lines.
                              If specified, will remove the common prefix read in each line.
        :return: Indices object
        """

        with open(path, "r") as f:
            lines = [l.strip() for l in f]
        if common_prefix is not None:
            lines = [pathlib.Path(l).relative_to(common_prefix) for l in lines]
        return Indices([str(l) for l in lines if l])

    @classmethod
    def from_dir(cls, path, suffix, relative=False, sort=False):
        """
        Create indices object by glob specific suffix in the given directory.

        :param path: the root path of files we want to create indices from
        :param suffix: the target file suffix, files are not with this suffix will be ignored.
        :param relative: if True, the resulting indices will be the relative path to `path`
        :return: Indices object
        """
        glob_pattern = f"*.{suffix.strip('.')}"
        paths = [p for p in pathlib.Path(path).rglob(glob_pattern)]
        if relative:
            paths = [p.relative_to(path) for p in paths]
        if sort:
            paths.sort()
        return Indices([str(p) for p in paths])

    def write(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self))


def get_indices(ind_path, root_dir: pathlib.Path, suffix, random_sample=None, seed=None):
    """
    Create file indices list either from indices_path or go through root_dir.
    :param ind_path: if not None, will return indices from this file, otherwise by traversing root_dir.
    :param root_dir: file root directory
    :param suffix: file suffix of target files
    :param random_sample: (optional) if given, will random sample this number of data and return.
    :param seed:
    :return: a list of path (str)
    """
    _sfx = suffix.strip(".")
    if ind_path:
        with open(ind_path, "r") as f:
            indices = [l.strip() for l in f.readlines()]
    else:
        indices = sorted([str(p.relative_to(root_dir).with_suffix("")) for p in root_dir.rglob(f"*.{_sfx}")])

    n_samples_to_draw = min(len(indices), random_sample or len(indices))

    if seed is not None:
        np.random.seed(seed)

    return np.random.choice(indices, size=n_samples_to_draw, replace=False).tolist()
