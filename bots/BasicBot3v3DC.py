import replay
import torch
import os
import datasets.dataset_utils as du

from bots import interactive
from models import Basic3v3DC
from replay import Team, Input


class BasicBot3v3DC(interactive.Interactive):
    def __init__(self, channel_id, name):
        super().__init__(channel_id)

        # Load pre-trained model and set-up the bot
        self.model = Basic3v3DC.Basic3v3DC()
        path = os.path.join(os.getcwd(), 'saved_models', name)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.last_inputs = []
        self.tick = 0

    def onUpdate(self):
        self.tick += 1
        if self.player and len(self.game.players) == 6 and self.tick % 3 == 0:
            # convert game state to tensor
            # tensor must be same format as how network was trained
            num_actions = 5
            our_team = self.player.team

            game_state = self.game

            # flip incoming states if on opposite side
            if self.player.team == Team.Blue:
                game_state = du.flip_state(game_state, x_axis_flip=False, y_axis_flip=True)

            # forming input only works for two players currently
            our_players = [p for p in game_state.players if p.team == our_team]
            opp_players = [p for p in game_state.players if p.team != our_team]

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

            state.extend([game_state.ball.x, game_state.ball.y, game_state.ball.vx, game_state.ball.vy])

            state_tensor = torch.tensor(state)

            # get output for model
            actions = self.model(state_tensor)
            # print(actions)
            actions = (actions > 0.5).tolist()
            # print(actions)

            player_actions = actions[player_idx * num_actions: (player_idx + 1) * num_actions]

            # send input actions
            inputs = [replay.Input(1 << idx) for idx, x in enumerate(player_actions) if x != 0]
            if self.player.team == Team.Blue:
                inputs = du.flip_action_list(al=inputs, x_axis_flip=False, y_axis_flip=True)

            if len(inputs) > 0:
                self.last_inputs = inputs
            else:
                self.last_inputs = self.to_ball()

            self.setInput(*inputs)
        elif self.player and len(self.game.players) == 6:
            self.setInput(*self.last_inputs)

    def to_ball(self):
        t = 15
        px = self.player.disc.x
        py = self.player.disc.y

        bx = self.game.ball.x
        by = self.game.ball.y

        inputs = []
        if abs(px - bx) > t:
            inputs.append(Input.Right if px < bx else Input.Left)
        if abs(py - by) > t:
            inputs.append(Input.Down if py < by else Input.Up)
        return inputs
