import torch.nn as nn
import os
import torch
import torch.optim as optim
import data
from collections import deque
from game import Game
import numpy as np
import random


class NN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize) -> None:
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, outputSize),
        )

    def forward(self, x):
        return self.stack(x)

    def save(self, filename="model.pth"):
        if not os.path.exists("./model"):
            os.makedirs("./model")
        filename = os.path.join("./model", filename)
        torch.save(self.state_dict(), filename)


class QTrainer:
    def __init__(self, lr, gamma, model) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        nextState = torch.tensor(nextState, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            nextState = torch.unsqueeze(nextState, 0)
            reward = torch.unsqueeze(reward, 0)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(state)):
            qNew = reward[idx]
            qNew = reward[idx] + self.gamma * torch.max(self.model(nextState[idx]))

            target[idx][torch.argmax(action[idx]).item()] = qNew

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Network:
    def __init__(self) -> None:
        self.gamma = data.gamma
        self.memory = deque(maxlen=100_000)
        self.model = NN(1, data.hiddenSize, 2)
        self.trainer = QTrainer(data.lr, self.gamma, self.model)
        self.decayStep = 0
        self.net = 0
        self.rand = 0

    def getState(self, game: Game):
        state = [game.angle]
        return np.array(state, dtype=float)

    def getMove(self, state):
        epsilon = data.minEpsilon + (data.maxEpsilon - data.minEpsilon) * np.exp(
            -data.decayRate * self.decayStep
        )

        final_move = [0, 0]
        if np.random.rand() < epsilon:
            choice = random.randint(0, 1)
            final_move[choice] = 1
            self.rand += 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
            final_move[move] = 1
            self.net += 1

        self.decayStep += 1
        return final_move

    def trainShort(self, state, action, reward, nextState):
        self.trainer.trainStep(state, action, reward, nextState)
        self.remember(
            state,
            action,
            reward,
            nextState,
        )

    def remember(self, state, action, reward, nextState):
        self.memory.append((state, action, reward, nextState))

    def trainLong(self):
        if len(self.memory) > data.batchSize:
            miniSample = random.sample(self.memory, data.batchSize)
        else:
            miniSample = self.memory

        states, actions, rewards, nextStates = zip(*miniSample)
        self.trainer.trainStep(
            np.array(states), np.array(actions), np.array(rewards), np.array(nextStates)
        )
