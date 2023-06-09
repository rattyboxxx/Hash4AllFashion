"""Mode utility."""
import torch
from . import check, param, meter, math, tracer, metrics


_COLORS = dict(
    Red="\033[91m",
    Green="\033[92m",
    Blue="\033[94m",
    Cyan="\033[96m",
    White="\033[97m",
    Yellow="\033[93m",
    Magenta="\033[95m",
    Grey="\033[90m",
    Black="\033[90m",
    Default="\033[0m",
)


def singleton(cls):
    """Decorator for singleton class.

    Usage:
    ------
        >>> @utils.singleton
        >>> class A(object):
        >>>    ...
        >>> x = A()
        >>> y = A()
        >>> assert id(x) == id(y)

    """
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


def colour(string, color="Green"):
    """Add color for string."""
    color = _COLORS.get(color.capitalize(), "Default")
    result = "{}{}{}".format(color, string, _COLORS["Default"])
    return result


def get_named_class(module):
    """Get the class member in module."""
    from inspect import isclass

    return {k: v for k, v in module.__dict__.items() if isclass(v)}


def get_named_function(module):
    """Get the class member in module."""
    from inspect import isfunction

    return {k: v for k, v in module.__dict__.items() if isfunction(v)}


def get_device(gpus=None):
    """Decide which device to use for data when given gpus.

    If use multiple GPUs, then data only need to stay in CPU.
    If use single GPU, then data must move to that GPU.

    Returns
    -------
    parallel: True if len(gpus) > 1
    device: if parallel or gpus is empty then device is cpu.
    """
    if not gpus:
        parallel = False
        device = torch.device("cpu")
        return parallel, device
    if len(gpus) > 1:
        parallel = True
        device = torch.device("cpu")
    else:
        parallel = False
        device = torch.device(gpus[0])
    return parallel, device


def to_device(data, device):
    """Move data to device."""

    error_msg = "data must contain tensors or lists; found {}"
    if isinstance(data, (list, tuple)):
        return tuple(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    raise TypeError((error_msg.format(type(data))))