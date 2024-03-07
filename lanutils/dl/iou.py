import torch


class EmptyInputsError(Exception):
    pass


def validate_inputs(a, b):
    if len(a) == 0 or len(b) == 0:
        raise EmptyInputsError
    if any([sa != sb for sa, sb in zip(a.shape, b.shape)]):
        raise ValueError(f"inputs must have identical shape, but {a.shape} and {b.shape} is given.")


def iou(a, b):
    try:
        validate_inputs(a, b)
    except EmptyInputsError:
        return torch.tensor([], dtype=torch.float).view(-1, 1)


def giou(a, b):
    try:
        validate_inputs(a, b)
    except EmptyInputsError:
        return torch.tensor([], dtype=torch.float).view(-1, 1)
