from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class FTHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Defaults
    bitfit: bool = False
    adapter_size: int = -1
    layer_list: str = "q,fc_in"
    batch_size: int = 64
    wd_power_law: tuple = None  # Scale weight decay by number of edits
