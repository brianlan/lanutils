import pathlib
import os
import shutil
import functools

def glob_with_suffixes(path_or_dir, supported_suffixes, recursive=True, list_hidden=False, sort=True, always_return_list=True):
    """Will list the file paths where either upper or lower case suffix is matched.
    :param path_or_dir: file root directory, if it's not a directory, no glob will be done.
    :param supported_suffixes: a list of suffix of target files
    :param recursive:
    :param list_hidden:
    :param sort:
    :return: a list of path (pathlib.Path)
    """
    path_or_dir = pathlib.Path(path_or_dir)
    if path_or_dir.is_dir():
        paths = []
        _glob_func = path_or_dir.rglob if recursive else path_or_dir.glob
        _glob_pattern = {f"*.{s.strip('.')}".lower() for s in supported_suffixes}
        _glob_pattern |= {p.upper() for p in _glob_pattern}
        for pattern in _glob_pattern:
            paths.extend(list(_glob_func(pattern)))
        if not list_hidden:
            paths = [p for p in paths if not p.name.startswith(".")]
    else:
        paths = [path_or_dir]
    if sort:
        paths = sorted(paths)
    if not always_return_list and len(paths) == 1:
        return paths[0]
    return paths


def delete_if_exist(file_path):
    file_path = str(file_path)
    try:
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)
    except FileNotFoundError:
        pass


def find_leaf_dirs(root_dir: pathlib.Path, leaf_dirs=None):
    """this function will find all the leaf directories recursively.
    :return: list of pathlib.Path objects
    """
    root_dir = pathlib.Path(root_dir)
    if leaf_dirs is None:
        leaf_dirs = []
    any_dir = False
    for p in os.listdir(root_dir):
        if (root_dir / p).is_dir():
            find_leaf_dirs(root_dir / p, leaf_dirs)
            any_dir = True
    if not any_dir:
        leaf_dirs.append(root_dir)
    return leaf_dirs


def ensured_path(input, ensure_parent=False):
    """Often used in the scenario that the path we want to write things to is ensured to be exist."""
    p = pathlib.Path(input)
    if ensure_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


parent_ensured_path = functools.partial(ensured_path, ensure_parent=True)
