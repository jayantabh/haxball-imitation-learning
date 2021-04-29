import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.fc1 = nn.Linear(12, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 20)
        self.fc4_a1 = nn.Linear(20, 1)
        self.fc4_a2 = nn.Linear(20, 1)
        self.fc4_a3 = nn.Linear(20, 1)
        self.fc4_a4 = nn.Linear(20, 1)
        self.fc4_a5 = nn.Linear(20, 1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        a1 = F.sigmoid(self.fc4_a1(x)).view(-1, 1)
        a2 = F.sigmoid(self.fc4_a2(x)).view(-1, 1)
        a3 = F.sigmoid(self.fc4_a3(x)).view(-1, 1)
        a4 = F.sigmoid(self.fc4_a4(x)).view(-1, 1)
        a5 = F.sigmoid(self.fc4_a5(x)).view(-1, 1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        outs = torch.cat((a1, a2, a3, a4, a5), 1)
        return outs
