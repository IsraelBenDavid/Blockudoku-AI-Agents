import csv
import os
import random

import numpy as np
import pygame as pg
import torch
import torch.optim as optim
from torch.distributions import Categorical

from PolicyNetwork import PolicyNetwork, state_transform, action_transform
from Constants import UNIFORM_PG_WEIGHTS_PATH
import Engine

EPSILON = 0.2
DISCOUNT = 0.7
LEARNING_RATE = 0.0001
NUM_EPISODES = 10000000
BATCH_SIZE = 100
ACTION_SIZE = 81

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        self.epsilon = EPSILON

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_legal_actions(self, game=None):
        if game is None:
            game = self.env
        actions_tuples = game.get_agent_legal_actions()
        actions = []
        for act in actions_tuples:
            actions.append((act[0] * 9) + act[1])
        return actions

    def get_action(self, game):  # for inference
        self.model.eval()
        legal_actions = self.get_legal_actions(game)
        state = state_transform(game.state).unsqueeze(0).to(device)
        with torch.inference_mode():
            action_probs = self.model(state)
        masked_action_probs = torch.zeros_like(action_probs)
        masked_action_probs[0][legal_actions] = action_probs[0][legal_actions]
        # masked_action_probs /= masked_action_probs.sum()
        action = torch.argmax(masked_action_probs).item()
        return action_transform(action)

    def action(self, state):  # epsilon greedy for training
        legal_actions = self.get_legal_actions()
        if len(legal_actions) == 0:
            return 0
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions)
        self.model.eval()
        state = state.unsqueeze(0).to(device)
        with torch.inference_mode():
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

    def generate_random_state(self, steps=1):
        self.env.reset()
        for i in range(steps):
            action = random.choice(self.get_legal_actions())
            self.env.apply_action(action_transform(action), render=False)
            random_op_action = random.choice(self.env.get_opponent_legal_actions())
            self.env.apply_opponent_action(random_op_action)

    def train(self, num_episodes, batch_size, render=False, save=False, save_path="", lr=LEARNING_RATE,
              csv_path=None):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if render:
            pg.init()
            screen = pg.display.set_mode([int(self.env.window_size.x), int(self.env.window_size.y)])
            self.env.setScreen(screen)
        invalids = []
        rewards = []
        scores = []
        steps_record = []
        for e in range(1, num_episodes + 1):
            state = self.env.state
            state = state_transform(state)
            tot_rewards = 0
            invalid_steps = 0
            total_steps = 0
            episode_memory = []
            while not self.env.lost:
                if total_steps == 100:  # steps limit
                    break
                total_steps += 1
                action = self.action(state)
                row, col = action_transform(action)
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
                state = state_transform(next_state)

                if render:
                    self.env.drawGameHeadless()
                    pg.display.set_caption(f"reward: {str(reward)} total: {tot_rewards:.3f} step: {total_steps}")

            self.memory.append(episode_memory)

            invalids.append(invalid_steps / total_steps)
            rewards.append(tot_rewards)
            scores.append(self.env.score)
            steps_record.append(total_steps)

            if e % batch_size == 0:
                self.update_policy()
                avg_invalids = sum(invalids) / len(invalids)
                avg_reward = sum(rewards) / len(rewards)
                avg_score = sum(scores) / len(scores)
                avg_steps = sum(steps_record) / len(steps_record)

                if csv_path:
                    # Data to append
                    data = [e / batch_size, f"{avg_steps: .4f}", f"{avg_invalids:.4f}", f"{avg_reward:.4f}",
                            f"{avg_score:.4f}"]

                    # Open the file in append mode ('a') and write the data
                    with open(csv_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(data)  # Write a single row to the CSV file

                print(f"-------------------{e / batch_size}---------"
                      f"mean steps = {avg_steps:.4f}"
                      f" | mean invalids = {avg_invalids:.4f}"
                      f" | mean reward = {avg_reward:.4f}"
                      f" | mean score = {avg_score:.4f}")
                invalids = []
                rewards = []
                scores = []
                steps_record = []

            self.env.reset()

            if save and e % 1000 == 0:
                self.save_model(save_path)
                print("saved")

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.to("cpu").state_dict(), filepath)
        self.model.to(device)

    def load_model(self, filepath):
        if os.path.isfile(filepath):
            self.model.to("cpu")
            self.model.load_state_dict(torch.load(filepath))
            self.model.to(device)
            print(f"Model loaded from {filepath}")
        else:
            print(f"File {filepath} does not exist. Cannot load the model.")


if __name__ == "__main__":
    print(f"Using {device} device")
    agent = PolicyGradientAgent(env=Engine.Blockudoku())

    # load the pretrained uniform distributed (across valid actions) network
    agent.load_model(UNIFORM_PG_WEIGHTS_PATH)

    # set discount factor and epsilon
    agent.set_gamma(DISCOUNT)
    agent.set_epsilon(EPSILON)

    # train
    agent.train(num_episodes=NUM_EPISODES,
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE,
                render=False,
                save=False,
                save_path="PATH/TO/SAVE",
                )
