import logging
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Union
import warnings
import os
from io import BytesIO

import torch
from typing import Collection
import os
import torch
import re
from collections import OrderedDict
from functools import cmp_to_key

















































































































def _get_checkpoint_paths(output_dir: str, last_n: int = 5):
    """
    Get the paths of the last 'last_n' checkpoints by parsing filenames
    in the output directory.
    """
    
    files = os.listdir(output_dir)
    
    checkpoint_files = [f for f in files if f.startswith("model.pt.e")]
    
    checkpoint_files.sort(
        key=lambda x: int(re.search(r"(\d+)", x).group()), reverse=True
    )
    
    checkpoint_paths = [os.path.join(output_dir, f) for f in checkpoint_files[:last_n]]
    return checkpoint_paths


@torch.no_grad()
def average_checkpoints(output_dir: str, last_n: int = 5):
    """
    Average the last 'last_n' checkpoints' model state_dicts.
    If a tensor is of type torch.int, perform sum instead of average.
    """
    checkpoint_paths = _get_checkpoint_paths(output_dir, last_n)
    state_dicts = []

    
    for path in checkpoint_paths:
        if os.path.isfile(path):
            state_dicts.append(torch.load(path, map_location="cpu")["state_dict"])
        else:
            print(f"Checkpoint file {path} not found.")
            continue

    
    if not state_dicts:
        raise RuntimeError("No checkpoints found for averaging.")

    
    avg_state_dict = OrderedDict()
    for key in state_dicts[0].keys():
        tensors = [state_dict[key].cpu() for state_dict in state_dicts]
        
        if str(tensors[0].dtype).startswith("torch.int"):
            
            summed_tensor = sum(tensors)
            avg_state_dict[key] = summed_tensor
        else:
            
            stacked_tensors = torch.stack(tensors)
            avg_state_dict[key] = torch.mean(stacked_tensors, dim=0)

    torch.save(
        {"state_dict": avg_state_dict},
        os.path.join(output_dir, f"model.pt.avg{last_n}"),
    )
    return avg_state_dict
