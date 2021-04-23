from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from replay import Replay, State
import random
import math

class BasicFiltered3v3(Dataset):
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
                                # self.game_states.append((state, 0))
                                # self.game_states.append((state, 1))
                                print(state.__dict__)
                                break

        self.filter_states()

    def filter_states(self):
        processed_states = []

        for state, player_team in self.game_states:
            our_player = state.players[0] if state.players[0].team == player_team else state.players[1]

            # Remove states where game is not being played e.g. Menu, Pause, Goal etc.
            if state.state != State.Game:
                continue

            # Get all states where the ball is kicked
            if our_player.input[4]:
                processed_states.append((state, player_team))
                continue

            # Randomization based filtering
            if sum(our_player.input) == 0 and random.random() < 0.8:
                processed_states.append((state, player_team))

            if sum(our_player.input) == 1 and random.random() < 0.5:
                processed_states.append((state, player_team))

            if sum(our_player.input) == 2 and random.random() < 0.5:
                processed_states.append((state, player_team))

            # Remove inconsistent game states
            if sum(our_player.input) == 3:
                if our_player.input[0] and our_player.input[1]:
                    continue

                if our_player.input[2] and our_player.input[3]:
                    continue

                processed_states.append((state, player_team))

        self.game_states = processed_states

    def __len__(self):
        return len(self.game_states)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        state, player_team = self.game_states[idx]

        print()

        sample = dict()
        sample['input'] = []

        our_player = state.players[0] if state.players[0].team == player_team else state.players[1]
        opp_player = state.players[0] if state.players[1].team == player_team else state.players[1]

        # Add player data to sample
        sample['input'].extend([our_player.disc.x, our_player.disc.y, our_player.disc.vx, our_player.disc.vy])
        sample['input'].extend([opp_player.disc.x, opp_player.disc.y, opp_player.disc.vx, opp_player.disc.vy])

        # Add ball data to sample
        sample['input'].extend([state.ball.x, state.ball.y, state.ball.vx, state.ball.vy])

        sample['input'] = torch.tensor(sample['input'])

        sample['output'] = torch.tensor(our_player.input)

        return sample['input'].float(), sample['output'].float()