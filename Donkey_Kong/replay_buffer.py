import random
import torch
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        # On limite la mémoire (ex: 10 000 ou 50 000 transitions)
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # On ajoute une expérience dans la pile
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # On pioche au hasard un groupe (batch) d'expériences pour l'entraînement
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)