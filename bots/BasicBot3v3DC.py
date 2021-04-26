import replay
import torch
import os

from bots import interactive
from models import Basic3v3DC


class BasicBot3v3DC(interactive.Interactive):
    def __init__(self, channel_id, name):
        super().__init__(channel_id)

        # Load pre-trained model and set-up the bot
        self.model = Basic3v3DC.Basic3v3DC()
        path = os.path.join(os.getcwd(), 'saved_models', name)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def onUpdate(self):
        if self.player and len(self.game.players) == 6:
            # convert game state to tensor
            # tensor must be same format as how network was trained
            num_actions = 5
            our_team = self.player.team

            # forming input only works for two players currently
            our_players = [p for p in self.game.players if p.team == our_team]
            opp_players = [p for p in self.game.players if p.team != our_team]

            our_players.sort(key=lambda p: (p.disc.x, p.disc.y))
            opp_players.sort(key=lambda p: (p.disc.x, p.disc.y))

            player_idx = 0
            for i, p in enumerate(our_players):
                if p.id == self.player.id:
                    player_idx = i
                    break

            state = []
            player_ = []
            teammates_ = []

            for i, player in enumerate(our_players):
                if i == player_idx:
                    player_.extend([player.disc.x, player.disc.y, player.disc.vx, player.disc.vy])
                else:
                    teammates_.extend([player.disc.x, player.disc.y, player.disc.vx, player.disc.vy])

            state.extend(player_)
            state.extend(teammates_)

            for player in opp_players:
                state.extend([player.disc.x, player.disc.y, player.disc.vx, player.disc.vy])

            state.extend([self.game.ball.x, self.game.ball.y, self.game.ball.vx, self.game.ball.vy])

            state_tensor = torch.tensor(state)

            # get output for model
            actions = self.model(state_tensor)
            actions = (actions > 0.5).tolist()

            player_actions = actions[player_idx * num_actions: (player_idx + 1) * num_actions]

            # send input actions
            inputs = [replay.Input(1 << idx) for idx, x in enumerate(player_actions) if x != 0]
            self.setInput(*inputs)
