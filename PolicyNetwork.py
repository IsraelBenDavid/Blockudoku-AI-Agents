import torch
import torch.nn.functional as F
import torch.nn as nn

N_BLOCKS = 3
device = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()

        # Initial convolutional block
        self.conv = nn.Sequential(
            nn.Conv2d(state_size[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.25),
        )

        # Residual blocks with convolutional layers
        self.blocks = nn.ModuleList()
        for _ in range(N_BLOCKS):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(p=0.25),
            ))

        # Policy network
        self.policy_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9 * 9 * 64, 2048),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, action_size)
        )

    def forward(self, state):
        x = self.conv(state)
        rec = x
        for block in self.blocks:
            out = block(rec)
            rec = out + rec
        action_logits = self.policy_fc(rec)
        action_probs = F.softmax(action_logits - action_logits.max(dim=-1, keepdim=True)[0], dim=-1)
        return action_probs


def state_transform(state):
    tensor = torch.from_numpy(state).float()
    tensor = tensor.permute(2, 0, 1)
    return tensor


def action_transform(action):
    row = action // 9
    col = action % 9
    return row, col
