import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Agent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=0.08):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.dqn = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.i = 0
        self.writer = SummaryWriter("logs")

    def __del__(self):
        self.writer.close()

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            print('----Random-----')
            return np.random.choice(self.action_size)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        q_values = self.dqn(state)
        return q_values.argmax().item()

    def log_scalar(self, log_name: str, scalar):
        self.writer.add_scalar(log_name, scalar, self.i)

    def train(self, state, action, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float).unsqueeze(0)

        q_values = self.dqn(state)
        next_q_values = self.dqn(next_state)
        q_value = q_values[0][action]

        if done:
            q_target = reward
        else:
            q_target = reward + self.discount_factor * next_q_values.max()

        loss = (q_value - q_target) ** 2
        self.writer.add_scalar("loss", loss, self.i)
        self.writer.add_scalar("reward", reward, self.i)
        self.i += 1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
