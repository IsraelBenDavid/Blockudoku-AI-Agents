import os
import random

import numpy as np
import pygame as pg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Categorical

import MinMaxEngine

device = "cuda" if torch.cuda.is_available() else "cpu"

N_BLOCKS = 5
DISCOUNT = 0.9
LEARNING_RATE = 2**-13
BATCH_SIZE = 50
ACTION_SIZE = 81


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()

        # Initial convolutional block with max-pooling layers
        self.conv = nn.Sequential(
            nn.Conv2d(state_size[0], 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Max-pooling after third Conv layer
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        # Common fully connected block
        self.blocks = nn.ModuleList()
        for i in range(N_BLOCKS):
            self.blocks.append(nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(p=0.25),
            ))

        # Policy network
        self.policy_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_size)
        )

        # Value network
        self.value_fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        x = self.conv(state)
        rec = x
        for i in range(N_BLOCKS):
            out = self.blocks[i](rec)
            rec = out + rec

        # Separate policy and value outputs
        action_logits = self.policy_fc(rec)
        # state_value = self.value_fc(rec)

        # Apply softmax to get action probabilities
        action_probs = F.softmax(action_logits - action_logits.max(), dim=-1)

        return action_probs


class PolicyGradientAgent:
    def __init__(self, env=None):
        self.env = env
        self.state_size = (2, 9, 9)
        if env is not None:
            self.state_size = (env.state.shape[2], env.state.shape[1], env.state.shape[0])
        self.action_size = ACTION_SIZE  # Actions

        self.model = PolicyNetwork(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.gamma = DISCOUNT
        self.memory = []
        self.flag = False

    def set_gamma(self, gamma):
        self.gamma = gamma


    def get_legal_actions(self, game=None):
        if game is None:
            game = self.env
        actions_tuples = game.get_agent_legal_actions()
        actions = []
        for act in actions_tuples:
            actions.append((act[0] * 9) + act[1])
        return actions

    def get_action(self, game): # for inference
        self.model.eval()
        legal_actions = self.get_legal_actions(game)
        state = self.state_transform(game.state).unsqueeze(0).to(device)
        with torch.inference_mode():
            action_probs = self.model(state)
        masked_action_probs = torch.zeros_like(action_probs)
        masked_action_probs[0][legal_actions] = action_probs[0][legal_actions]
        # masked_action_probs /= masked_action_probs.sum()
        action = torch.argmax(masked_action_probs).item()
        return self.action_transform(action)

    def action(self, state):  # for training
        legal_actions = self.get_legal_actions()
        if len(legal_actions) == 0:
            return 0
        if np.random.rand() <= 0.4:
            return random.choice(legal_actions)
        self.model.eval()
        state = state.unsqueeze(0).to(device)
        action_probs = self.model(state)

        return torch.multinomial(action_probs, 1).item()

    def store(self, state, action, reward):
        self.memory.append((state, action, reward))

    def update_policy(self):
        returns = []
        actions = torch.empty(0)
        states = torch.empty(0)

        # Compute the discounted rewards for each episode
        for mem in reversed(self.memory):
            R = 0
            for _, _, reward in reversed(mem):
                R = reward + self.gamma * R
                returns.insert(0, R)

            # Unpacking the current states and actions from memory
            curr_states, curr_actions, _ = zip(*mem)

            # Stack and concatenate with existing tensors
            curr_states_tensor = torch.stack(curr_states)
            curr_actions_tensor = torch.tensor(curr_actions)

            if states.numel() == 0:  # If the states tensor is empty
                states = curr_states_tensor
            else:
                states = torch.cat((curr_states_tensor, states), dim=0)

            if actions.numel() == 0:  # If the actions tensor is empty
                actions = curr_actions_tensor
            else:
                actions = torch.cat((curr_actions_tensor, actions), dim=0)

        # Convert to tensors and normalize returns
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        states = states.to(device)
        actions = actions.to(device)

        # Train policy network
        self.model.train()

        # Get the action probabilities from the model
        action_probs = self.model(states)
        action_dist = Categorical(action_probs)

        # Calculate log probabilities of the actions taken
        log_probs_new = action_dist.log_prob(actions)

        # Policy loss (negative of the sum of log probabilities weighted by returns)
        policy_loss = -(log_probs_new * returns).sum()

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear the memory after the update
        self.memory = []


    def state_transform(self, state):
        tensor = torch.from_numpy(state).float()
        tensor = tensor.permute(2, 0, 1)
        return tensor

    def action_transform(self, action):
        row = action // 9
        col = action % 9
        return row, col

    def print_state(self, state):
        print("board:")
        for row in state[0]:
            for cell in row:
                if cell.item() == 1:
                    print("*", end=" ")
                else:
                    print(" ", end=" ")
            print()
        print("shape:")
        for row in state[1]:
            for cell in row:
                if cell.item() == 1:
                    print("*", end=" ")
                else:
                    print(" ", end=" ")
            print()

    def generate_random_state(self, steps=1):
        self.env.reset()
        for i in range(steps):
            action = random.choice(self.get_legal_actions())
            self.env.apply_action(self.action_transform(action), render=False)
            random_op_action = random.choice(self.env.get_opponent_legal_actions())
            self.env.apply_opponent_action(random_op_action)

    def train(self, num_episodes, batch_size, t, render=False, save=False, save_path=""):
        if render:
            pg.init()
            screen = pg.display.set_mode([int(self.env.window_size.x), int(self.env.window_size.y)])
            self.env.setScreen(screen)
        invalids = []
        rewards = []
        scores = []
        for e in range(1, num_episodes + 1):
            print(f"----------------------------------------- episode = {e}")
            state = self.env.state
            state = self.state_transform(state)
            tot_rewards = 0
            invalid_steps = 0
            total_steps = 0
            episode_memory = []
            while not self.env.lost:
                if total_steps == 30:  # steps limit
                    break
                total_steps += 1
                action = self.action(state)
                row, col = self.action_transform(action)
                old_score = self.env.score
                next_state, reward, done, valid = self.env.apply_action((row, col), render=render)
                reward = self.env.score - old_score
                if valid:
                    random_op_action = random.choice(self.env.get_opponent_legal_actions())
                    next_state = self.env.apply_opponent_action(random_op_action)
                    reward *= 10
                else:
                    invalid_steps += 1
                    reward = 0
                tot_rewards += reward
                episode_memory.append((state, action, reward))  # store
                state = self.state_transform(next_state)

                if render:
                    self.env.drawGameHeadless()
                    pg.display.set_caption(f"reward: {str(reward)} total: {tot_rewards:.3f} step: {total_steps}")

            self.memory.append(episode_memory)

            print(f"episode: {e}/{num_episodes}\n"
                  f"score: {self.env.score}\n"
                  f"reward: {tot_rewards}\n"
                  f"steps: {total_steps}\n"
                  f"invalid steps: {invalid_steps / total_steps}\n")

            if e % batch_size == 0:
                self.update_policy()

            invalids.append(invalid_steps / total_steps)
            rewards.append(tot_rewards)
            scores.append(self.env.score)
            if e % batch_size == 0:
                avg_invalids = sum(invalids) / len(invalids)
                avg_reward = sum(rewards) / len(rewards)
                avg_score = sum(scores) / len(scores)
                print(
                    f"------------------------------------------------------------------- mean invalids = {avg_invalids:.4f}"
                    f" | mean reward = {avg_reward:.3f} | mean score = {avg_score:.3f}")
                invalids = []

            self.env.reset()

            if save and e % 100 == 0:
                self.save_model(save_path)
                print("saved")


    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        if os.path.isfile(filepath):
            self.model.load_state_dict(torch.load(filepath))
            self.model.to(device)
            print(f"Model loaded from {filepath}")
        else:
            print(f"File {filepath} does not exist. Cannot load the model.")


if __name__ == "__main__":
    print(f"Using {device} device")
    game = MinMaxEngine.Blockudoku()
    agent = PolicyGradientAgent(game)

    lesson = 0
    gamma = 0.00
    agent.load_model("checkpoints/pg_unif/pg_unif.pth")
    while gamma < 1:
        print(f"********************************************************************************************")
        print(f"***************************************** Lesson {lesson} *****************************************")
        print(
            f"********************************************************************************************")
        agent.set_gamma(gamma)
        agent.train(2501, BATCH_SIZE, t=1, render=False, save=True,
                    save_path=f"checkpoints/pg_agent/pg_unif_finetuned_lesson_{lesson}.pth")
        lesson += 1
        gamma += 0.05



