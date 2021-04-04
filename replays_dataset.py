from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from replay import Replay


class HaxballDemoDataset(Dataset):
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
                            if state.players is not None and len(state.players) == 2:
                                self.game_states.append((state, 0))
                                self.game_states.append((state, 1))

    def __len__(self):
        return len(self.game_states)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        state, player_team = self.game_states[idx]

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


if __name__ == "__main__":
    dataset = HaxballDemoDataset('preprocessed/')

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)

