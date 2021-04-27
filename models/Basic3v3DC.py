import torch
import torch.nn as nn

class Basic3v3DC(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.model = nn.Sequential(
            nn.Linear(28, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 5),
            nn.Sigmoid(),
        )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.model(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs