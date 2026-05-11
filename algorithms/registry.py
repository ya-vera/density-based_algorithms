from .base import AlgorithmRegistry

from . import dbscan_wrapper
from . import hdbscan_wrapper
from . import optics_wrapper
from . import dpc_wrapper
from . import dac_wrapper
from . import rd_dac_wrapper
from . import ckdpc_wrapper

__all__ = ["AlgorithmRegistry", "get_algorithm", "list_algorithms"]


def get_algorithm(name: str):
    return AlgorithmRegistry.get(name.lower())


def list_algorithms():
    return AlgorithmRegistry.list_algorithms()


def get_all_algorithms():
    return AlgorithmRegistry._algorithms.copy()