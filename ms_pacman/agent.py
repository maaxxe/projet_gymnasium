import torch
import torch.optim as optim
import random
from dqn_self import DQNCerveau
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, n_actions, device, agent_type="pacman"):
        self.n_actions = n_actions
        self.device = device
        self.agent_type = agent_type

        self.policy_net = DQNCerveau(n_actions).to(device)
        self.target_net = DQNCerveau(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(50000)

        # Les fantômes explorent différemment de Pac-Man
        self.epsilon = 1.0
        self.epsilon_min = 0.05 if agent_type == "ghost" else 0.1
        self.epsilon_decay = 0.9995

        self.power_pellet_active = False

        def set_power_pellet(self, active: bool):
            """Appelé depuis run.py quand raw_reward == 50 (power pellet mangé)."""
            self.power_pellet_active = active

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_t = torch.tensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            expected_q = rewards + (0.99 * next_q * (1 - dones))
        # Huber loss à la place de MSE → plus stable avec des grandes rewards shapées
        loss = torch.nn.functional.smooth_l1_loss(current_q.squeeze(), expected_q)
        #loss = torch.nn.functional.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping ajouté pour éviter les explosions de gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
