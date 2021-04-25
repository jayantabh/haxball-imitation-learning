from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from replay import Replay, State
import random
import math

class Basic3v3DC(Dataset):
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
                if name.split('.')[-1] != "bin":
                    continue

                with open(os.path.join(self.root_dir, name), 'rb') as f:
                    file_content = f.read()
                    _, states = Replay(file_content)

                    for state in states:
                        if state.players is None:
                            continue

                        if len(state.players) != 6:
                            continue

                        our_players = state.players[:3]
                        opp_players = state.players[3:]

                        our_players.sort(key=lambda p: (p.disc.x, p.disc.y))
                        opp_players.sort(key=lambda p: (p.disc.x, p.disc.y))

                        for i in range(len(our_players)):
                            inputs = []
                            player_ = []
                            teammates_ = []

                            for j, player in enumerate(our_players):
                                if i == j:
                                    player_.extend([player.disc.x, player.disc.y, player.disc.vx, player.disc.vy])
                                else:
                                    teammates_.extend([player.disc.x, player.disc.y, player.disc.vx, player.disc.vy])

                            inputs.extend(player_)
                            inputs.extend(teammates_)

                            for player in opp_players:
                                inputs.extend([player.disc.x, player.disc.y, player.disc.vx, player.disc.vy])

                            inputs.extend([state.ball.x, state.ball.y, state.ball.vx, state.ball.vy])

                            outputs = [*our_players[i].input]

                            self.game_states.append(
                                {
                                    "inputs": inputs,
                                    "outputs": outputs
                                }
                            )

    def __len__(self):
        return len(self.game_states)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        nn_inputs = torch.tensor(self.game_states[idx]["inputs"])

        nn_outputs = torch.tensor(self.game_states[idx]["outputs"])

        return nn_inputs.float(), nn_outputs.float()
