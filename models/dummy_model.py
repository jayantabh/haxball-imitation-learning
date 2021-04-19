import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        # Dummy Model

    def forward(self, x):
        print(x.shape)
        return torch.tensor(
            [
                True, False, True, False, False,
                False, True, False, True, False,
                False, False, True, False, True
            ]
        )
