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

print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# N_BLOCKS = 3 # 1
N_BLOCKS = 5  # 2


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.first_fc_block = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(N_BLOCKS):
            self.blocks.append(nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU()
            ))
        self.action_block = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
        self.action_fc = nn.Linear(128, action_size)
        self.v_fc = nn.Linear(128, 1)

    def forward(self, state):
        x = self.first_fc_block(state)
        rec = x
        for i in range(N_BLOCKS):
            out = self.blocks[i](rec)
            rec = out + rec
        rec = self.action_block(rec)
        action = self.action_fc(rec)
        v = self.v_fc(rec)
        return action + v


STATE_SIZE = 162
ACTION_SIZE = 5


class QLAgent:
    def __init__(self, env):
        self.env = env
        self.qnetwork = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_qnet = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_qnet.load_state_dict(self.qnetwork.state_dict())
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)  # Implement replay buffer here
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.total_loss = 0

    def choose_action(self, state, test=False):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon and not test:
            return random.choice(range(ACTION_SIZE))

        self.qnetwork.eval()
        # state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Ensure the input is 2D (1, state_size)
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            q_values = self.qnetwork(state.to(device))
        action = q_values.argmax().item()

        return action

    def update_policy(self, batch_size):
        # Sample a batch from memory
        batch = random.sample(self.memory, batch_size)
        # batch = self.memory
        states = torch.tensor(np.array([s for s, _, _, _, _ in batch])).to(device).float()
        actions = torch.tensor([a for _, a, _, _, _ in batch]).to(device).float()
        rewards = torch.tensor([r for _, _, r, _, _ in batch]).to(device).float()
        next_states = torch.tensor(np.array([ns for _, _, _, ns, _ in batch])).to(device).float()
        dones = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32).to(device).float()

        # Compute the target Q-value
        with torch.no_grad():
            next_q_values = self.target_qnet(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        self.qnetwork.train()
        # Get current Q-value
        # old_q_values = self.qnetwork(states).gather(1, actions.unsqueeze(1)).squeeze()
        old_q_values = self.qnetwork(states).gather(1, actions.long().unsqueeze(1)).squeeze()

        # Compute loss and optimize
        loss = F.mse_loss(old_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_loss += loss.item()
        # print("loss = ", loss)

    def train(self, episodes, batch_size, save=False, render=True):
        n_steps = 0

        for episode in range(episodes):
            print(f"----------------------------------------- episode = {episode}")
            self.env.reset()
            state = self.env.state
            state = state.flatten()

            game_over = False
            total_reward = 0
            curr_step = 0

            while not game_over:
                n_steps += 1
                curr_step += 1
                action = self.choose_action(state)
                next_state, reward, game_over = self.env.step(action)
                self.memory.append((state, action, reward, next_state.flatten(), game_over))
                state = next_state.flatten()

                total_reward += reward

                if render:
                    self.env.drawGameHeadless()
                    pg.display.set_caption(f"reward: {str(reward)} "
                                               f"epsilon: {self.epsilon:.3f} "
                                               f"total: {total_reward:.3f} "
                                           f"step: {curr_step}")
                self.epsilon = max(0.01, self.epsilon * 0.99)
                # self.epsilon = 0
                if len(self.memory) > batch_size and n_steps % batch_size == 0:
                    # print("--- training")
                    self.total_loss = 0
                    trains_iters = 10
                    for _ in range(trains_iters):
                        self.update_policy(batch_size)
                    self.total_loss /= trains_iters
                    # print(f"--- done training \n mean loss = {self.total_loss}")


                if n_steps % (batch_size * 3) == 0 and save:
                    self.save_model("checkpoints/qmodel_2.pth")

                if n_steps % (batch_size * 6) == 0:
                    self.epsilon = 1.

                if curr_step == 1000 or game_over:
                    break
            if game_over:
                print("LOST THE GAME")
            print(f"total score = {self.env.score}")
            print("total reward = ", total_reward)
            print(f"last loss = {self.total_loss}")
            self.tau = 1.
            for target_network_param, q_network_param in zip(self.target_qnet.parameters(),
                                                             self.qnetwork.parameters()):
                target_network_param.data.copy_(
                    self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                )
            # Decay epsilon to reduce exploration over time


    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.qnetwork.state_dict(), filepath)
        # print("saved")

    def load_checkpoint(self, filepath):
        if os.path.isfile(filepath):
            self.qnetwork.load_state_dict(torch.load(filepath))
            # self.qnetwork.eval()  # Set the model to evaluation mode
            print("Model loaded successfully from {}".format(filepath))
        else:
            print("No checkpoint found at {}".format(filepath))


import os
if __name__ == "__main__":
    load = False
    render = False
    save = True
    batch_size = 256
    episodes = 10000000

    game = Engine.Blockudoku()
    if render:
        pg.init()
        screen = pg.display.set_mode([int(game.window_size.x), int(game.window_size.y)])
        game.setScreen(screen)

    agent = QLAgent(game)
    if load:
        agent.load_checkpoint("checkpoints/qmodel.pth")

    agent.train(episodes, batch_size, save=save, render=render)
