"""Initialize funasr package."""

import os
import pkgutil
import importlib

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "version.txt")
with open(version_file, "r") as f:
    __version__ = f.read().strip()


import importlib
import pkgutil


def import_submodules(package, recursive=True):
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            results[name] = importlib.import_module(name)
        except Exception as e:
            
            
            pass
        if recursive and is_pkg:
            results.update(import_submodules(name))
    return results


import_submodules(__name__)

from src.models.src_step_audio.funasr_detach.auto.auto_model import AutoModel
from src.models.src_step_audio.funasr_detach.auto.auto_frontend import AutoFrontend
