import replay
import torch
import os

from bots import interactive
from models import DummyModel


class MyModelBot(interactive.Interactive):
    def __init__(self, channel_id):
        super().__init__(channel_id)

        # Load pre-trained model and set-up the bot
        self.model = DummyModel()

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
            for player in our_players:
                state.extend([player.disc.x, player.disc.y, player.disc.vx, player.disc.vy])

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