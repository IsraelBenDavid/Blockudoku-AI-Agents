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
from baseline_agent import BaselineAgent
from q_learning_agent import QLAgent

if __name__ == "__main__":
    game = Engine.Blockudoku()

    pg.init()

    screen = pg.display.set_mode([int(game.window_size.x), int(game.window_size.y)])

    # game.seed(69)
    game.setScreen(screen)

    agent = BaselineAgent(game, 1, 0.001, 0.995)
    agent.load_model("checkpoints/baseline/model.pth")

    # agent = QLAgent(game, 1, 0.001, 0.995)
    # agent.load_model("checkpoints/ql_agent/model.pth")

    # game.render()
    game_over = False
    running = True
    while not game_over and running:
        time.sleep(0.5)
        state = agent.state_transform(game.state)
        action = agent.test_action(state)
        next_state, reward, game_over = game.step(action)
        running = game.drawGameHeadless()
        pg.display.set_caption(f"reward: {str(reward)}")

    pg.quit()
