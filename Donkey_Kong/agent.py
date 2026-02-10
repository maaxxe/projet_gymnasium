import torch
import torch.optim as optim
import numpy as np
import random
from dqn_self import DQNCerveau
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, n_actions, device):
        self.n_actions = n_actions
        self.device = device
        
        # Le cerveau qui apprend
        self.policy_net = DQNCerveau(n_actions).to(device)
        # Le cerveau "cible" (pour stabiliser l'apprentissage)
        self.target_net = DQNCerveau(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        #"Le Policy Net apprend en temps réel, mais pour calculer l'objectif à atteindre sans que l'IA ne devienne instable,
        #  on utilise le Target Net qui est une copie 'figée'. On ne le met à jour que tous les 
        # 1000 pas pour que la cible ne bouge pas trop vite."
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(10000) # La mémoire qu'on a codée avant
        
        self.epsilon = 1.0  # 100% de hasard au début
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99995 # Diminue très lentement

    def select_action(self, state):
        # Stratégie Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # On pioche dans la mémoire
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Calcul de la valeur Q actuelle
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Calcul de la valeur Q attendue (équation de Bellman)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            expected_q = rewards + (0.99 * next_q * (1 - dones))

        # Calcul de l'erreur (Loss) et mise à jour
        loss = torch.nn.functional.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Réduction du hasard
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay