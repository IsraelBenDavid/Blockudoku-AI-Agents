import os
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

DISCOUNT = 0.95
LEARNING_RATE = 0.001
BATCH_SIZE = 24
STATE_SIZE = 162
ACTION_SIZE = 5


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(state_size, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 80),
        #     nn.ReLU(),
        #     nn.Linear(80, 69),
        #     nn.ReLU(),
        #     nn.Linear(69, 40),
        #     nn.ReLU(),
        #     nn.Linear(40, 30),
        #     nn.ReLU(),
        #     nn.Linear(30, 24),
        #     nn.ReLU(),
        #     nn.Linear(24, action_size)
        # )
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=state_size[0], out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * (state_size[1] // 4) * (state_size[2] // 4), 69),  # Adjust dimensions according to pooling
            nn.ReLU(),
            nn.Linear(69, 69),
            nn.ReLU(),
            nn.Linear(69, action_size)
        )

    def forward(self, state):
        return self.model(state)


class BaselineAgent:
    def __init__(self, env, epsilon, epsilon_min, epsilon_decay):
        self.env = env
        self.state_size = (env.state.shape[2], env.state.shape[1], env.state.shape[0])
        self.action_size = env.action_space.n  # Actions

        self.memory = deque([], maxlen=2500)
        self.alpha = LEARNING_RATE
        self.gamma = DISCOUNT
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = QNetwork(self.state_size, self.action_size).to(device)
        self.model_target = QNetwork(self.state_size, self.action_size).to(device)  # Second (target) neural network
        self.update_target_from_model()  # Update weights
        self.loss = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def update_target_from_model(self):
        for target_network_param, q_network_param in zip(self.model_target.parameters(),
                                                         self.model.parameters()):
            target_network_param.data.copy_(q_network_param.data)

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        state = state.unsqueeze(0).to(device)
        action_vals = self.model(state)  # Exploit: Use the NN to predict the correct action from this state
        return torch.argmax(action_vals[0])

    def test_action(self, state):  # Exploit
        state = state.unsqueeze(0).to(device)
        action_vals = self.model(state)  # Exploit: Use the NN to predict the correct action from this state
        return torch.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, nstate, done))

    def experience_replay(self, batch_size):
        # Assuming minibatch is a list of tuples: (state, action, reward, next_state, done)
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors and move to the correct device
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones).float().to(device)

        # Predictions for current and next states
        st_predict = self.model(states)
        nst_predict = self.model(next_states)
        nst_predict_target = self.model_target(next_states)

        # Double DQN: Using the online model to choose the best action for next state
        nst_action_predict_model = nst_predict.max(1)[1]

        # Calculate target values
        targets = rewards + (1 - dones) * self.gamma * nst_predict_target.gather(1, nst_action_predict_model.unsqueeze(
            1)).squeeze()

        # Clone st_predict to avoid inplace operations
        y = st_predict.clone()
        y[range(batch_size), actions] = targets

        # Compute the loss and perform backpropagation
        loss = F.mse_loss(st_predict, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss.append(loss.item())

        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def state_transform(self, state):
        # Reshape the state to add a batch dimension if necessary
        # temp = np.reshape(state, (self.state_size[0], self.state_size[1], self.state_size[2]))

        # Convert the NumPy array to a PyTorch tensor
        tensor = torch.from_numpy(state).float()

        # Permute the dimensions from (height, width, channels) to (channels, height, width)
        tensor = tensor.permute(2, 0, 1)

        # Add the batch dimension
        # tensor = tensor.unsqueeze(0)

        return tensor

    def train(self, num_episodes, time_play, render=False, save=False):
        rewards = []  # Store rewards for graphing
        epsilons = []  # Store the Explore/Exploit
        if render:
            pg.init()
            screen = pg.display.set_mode([int(self.env.window_size.x), int(self.env.window_size.y)])
            self.env.setScreen(screen)

        for e in range(num_episodes):
            print(f"----------------------------------------- episode = {e}")
            state = self.env.state
            state = self.state_transform(state)  # Resize to store in memory to pass to .predict
            tot_rewards = 0
            done = False
            for time in range(
                    time_play):  # 200 is when you "solve" the game. This can continue forever as far as I know
                action = self.action(state)
                nstate, reward, done = self.env.step(action)
                nstate = self.state_transform(nstate)
                tot_rewards += reward
                self.store(state, action, reward, nstate, done)  # Resize to store in memory to pass to .predict
                state = nstate

                if render:
                    self.env.drawGameHeadless()
                    pg.display.set_caption(f"reward: {str(reward)} "
                                           f"epsilon: {self.epsilon:.3f} "
                                           f"total: {tot_rewards:.3f} "
                                           f"step: {time}")

                if done or time == time_play:
                    rewards.append(tot_rewards)
                    epsilons.append(self.epsilon)
                    print("episode: {}/{}, score: {}, e: {}"
                          .format(e, num_episodes, tot_rewards, self.epsilon))
                    break
                # Experience Replay
                if len(self.memory) > BATCH_SIZE:
                    self.experience_replay(BATCH_SIZE)
            if done:
                print("LOST THE GAME")
            print(f"total score = {self.env.score}")
            print("total reward = ", tot_rewards)
            print(f"last loss = {self.loss[-1]}")
            self.env.reset()
            # Update the weights after each episode (You can configure this for x steps as well
            self.update_target_from_model()
            if save:
                self.save_model("checkpoints/baseline/model.pth")

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        # Ensure the file exists before loading
        if os.path.isfile(filepath):
            self.model.load_state_dict(torch.load(filepath))
            self.model.to(device)  # Move the model to the appropriate device (CPU/GPU)
            print(f"Model loaded from {filepath}")
        else:
            print(f"File {filepath} does not exist. Cannot load the model.")


if __name__ == "__main__":
    game = Engine.Blockudoku()
    agent = BaselineAgent(game, 1, 0.001, 0.995)
    agent.train(100000, 1000, render=False, save=True)
