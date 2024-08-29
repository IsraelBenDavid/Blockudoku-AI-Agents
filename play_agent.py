import random
import sys
import time

import numpy as np
import pygame as pg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

import Engine
from q_learning_agent import QLAgent


if __name__ == "__main__":
    game = Engine.Blockudoku()

    pg.init()

    screen = pg.display.set_mode([int(game.window_size.x), int(game.window_size.y)])

    # game.seed(69)
    game.setScreen(screen)

    agent = QLAgent(game)
    agent.load_checkpoint("checkpoints/qmodel_2.pth")

    # game.render()
    game_over = False
    running = True
    while not game_over and running:
        # time.sleep(0.1)

        action = agent.choose_action(game.state.flatten(), test=True)
        next_state, reward, game_over = game.step(action)
        running = game.drawGameHeadless()
        pg.display.set_caption(f"reward: {str(reward)}")

    pg.quit()