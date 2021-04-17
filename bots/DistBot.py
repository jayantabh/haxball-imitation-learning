import replay
import torch
import os

from bots import interactive
from models import dist_model

import math

class DistBot(interactive.Interactive):

  def __init__(self, channel_id, name):
    super().__init__(channel_id)

    # Load pre-trained model and set-up the bot
    self.model = dist_model.MyModel()
    path = os.path.join(os.getcwd(), 'saved_models', name)
    self.model.load_state_dict(torch.load(path))
    self.model.eval()

  def onUpdate(self):
    if self.player and len(self.game.players) == 2:
      # convert game state to tensor
      # tensor must be same format as how network was trained

      # forming input only works for two players currently
      our_player = self.player
      state = [self.player.disc.x, self.player.disc.y, self.player.disc.vx, self.player.disc.vy]
      for player in self.game.players:
        if player.id != self.player.id:
          state.extend([player.disc.x, player.disc.y, player.disc.vx, player.disc.vy])
          opp_player = player

      state.extend([self.game.ball.x, self.game.ball.y, self.game.ball.vx, self.game.ball.vy])


      state.append(math.dist((our_player.disc.x, our_player.disc.y), (self.game.ball.x, self.game.ball.y)))
      state.append(math.dist((opp_player.disc.x, opp_player.disc.y), (self.game.ball.x, self.game.ball.y)))

      state_tensor = torch.tensor(state)

      # get output for model
      actions = self.model(state_tensor)
      actions = (actions > 0.5).tolist()
      
      # send input actions
      inputs = [replay.Input(1 << idx) for idx,x in enumerate(actions) if x != 0]
      self.setInput(*inputs)