import replay
import torch
import os
import datasets.dataset_utils as du

from bots import interactive
from replay import Team, Input


class GoTowardsBot3v3(interactive.Interactive):
    def __init__(self, channel_id):
        super().__init__(channel_id)

    def onUpdate(self):
        if self.player and len(self.game.players) == 6:

            # send input actions
            inputs = self.to_ball()

            self.setInput(*inputs)

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
