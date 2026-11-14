"""
Dataset Loaders module for ToTTo, WebNLG 2020, and DART.
"""
from .base_loader import BaseLoader, RawRecord
from .totto_loader import ToTToloader
from .webnlg_loader import WebNLGloader
from .dart_loader import DARTloader
from .loader_factory import LoaderFactory

__all__ = [
    "BaseLoader",
    "RawRecord",
    "ToTToloader",
    "WebNLGloader",
    "DARTloader",
    "LoaderFactory",
]
