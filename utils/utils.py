from pathlib import Path
import torch.nn as nn


def create_dir_if_not_exist(path):
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
