from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from replay import Replay, State
import random
import math

class Basic3v3(Dataset):
    """Haxball Demonstrations dataset."""

    def __init__(self, bin_file_path):
        """
        Args:
            bin_file_path (string): File path containing preprocessed
        """
        self.game_states = []
        self.root_dir = bin_file_path

        for root, dirs, files in os.walk(bin_file_path):
            for name in files:
                if name.split('.')[-1] == "bin":
                    with open(os.path.join(self.root_dir, name), 'rb') as f:
                        file_content = f.read()
                        _, states = Replay(file_content)

                        for state in states:
                            if state.players is not None and len(state.players) == 6:
                                self.game_states.append(state)

    def __len__(self):
        return len(self.game_states)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        state = self.game_states[idx]

        our_players = state.players[:3]
        opp_players = state.players[3:]

        our_players.sort(key=lambda p: (p.disc.x, p.disc.y))
        opp_players.sort(key=lambda p: (p.disc.x, p.disc.y))

        nn_inputs = []
        nn_outputs = []

        # Add player data to sample
        for player in our_players + opp_players:
            nn_inputs.extend([player.disc.x, player.disc.y, player.disc.vx, player.disc.vy])

        # Add ball data to sample
        nn_inputs.extend([state.ball.x, state.ball.y, state.ball.vx, state.ball.vy])

        nn_inputs = torch.tensor(nn_inputs)

        for player in our_players:
            nn_outputs.extend([*player.input])

        nn_outputs = torch.tensor(nn_outputs)       

        return nn_inputs.float(), nn_outputs.float()