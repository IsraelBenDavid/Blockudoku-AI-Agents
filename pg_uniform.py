import os
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
import MinMaxEngine
from policy_gradient_agent import PolicyNetwork

LEARNING_RATE = 0.001
BATCH_SIZE = 25
ACTION_SIZE = 81

device = "cuda" if torch.cuda.is_available() else "cpu"


class PGUniformAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = (env.state.shape[2], env.state.shape[1], env.state.shape[0])
        self.action_size = ACTION_SIZE
        self.model = PolicyNetwork(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = []

    def update_policy(self):
        self.model.train()

        states, actions = zip(*self.memory)
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)

        action_probs = self.model(states)

        eps = 1e-15
        action_probs = action_probs + eps  # to avoid log(0)

        uniform_loss = F.kl_div(action_probs.log(), actions, reduction='batchmean')

        self.optimizer.zero_grad()
        uniform_loss.backward()
        self.optimizer.step()

        self.memory = []
        return uniform_loss.item()

    def state_transform(self, state):
        tensor = torch.from_numpy(state).float()
        tensor = tensor.permute(2, 0, 1)
        return tensor

    def action_transform(self, action):
        row = action // 9
        col = action % 9
        return row, col

    def get_legal_actions(self):
        actions_tuples = self.env.get_agent_legal_actions()
        if len(actions_tuples) == 0: return []
        actions = torch.zeros(ACTION_SIZE)
        for act in actions_tuples:
            actions[(act[0] * 9) + act[1]] = 1
        return actions / actions.sum()

    def get_legal_actions_list(self):
        actions_tuples = self.env.get_agent_legal_actions()
        actions = []
        for act in actions_tuples:
            actions.append((act[0] * 9) + act[1])
        return actions

    def generate_random_state(self, steps=1):
        self.env.reset()
        for i in range(steps):
            if len(self.get_legal_actions_list()) == 0: return
            action = random.choice(self.get_legal_actions_list())
            self.env.apply_action(self.action_transform(action), render=False)
            random_op_action = random.choice(self.env.get_opponent_legal_actions())
            self.env.apply_opponent_action(random_op_action)

    def train(self, epochs, batch_size, save=False, save_path=""):
        losses = []
        for e in range(epochs):
            b = 0
            while b < batch_size:
                random_step = torch.randint(low=0, high=30, size=(1,)).item()
                self.generate_random_state(steps=random_step)
                legal_actions = self.get_legal_actions()
                if len(legal_actions) == 0:
                    continue
                b += 1
                state = self.state_transform(self.env.state)
                self.memory.append((state, legal_actions))

            loss_val = self.update_policy()
            losses.append(loss_val)

            print(f"epoch: {e + 1}/{epochs}\n"
                  f"loss_val: {loss_val}\n")

            if (e + 1) % 100 == 0:
                mean_loss = torch.tensor(losses).mean()
                print(f"---------------------------------------------------------------------- mean loss = {mean_loss}")
                losses = []

            if save and (e + 1) % 100 == 0:
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
    agent = PGUniformAgent(game)
    agent.train(100000000, BATCH_SIZE, save=True, save_path="checkpoints/pg_unif/pg_unif.pth")
