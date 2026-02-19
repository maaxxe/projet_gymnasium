import torch
from mspacman_env import make_env  # Ton wrapper custom
from agent import DQNAgent

env = make_env()
agent = DQNAgent(env.action_space.n, 'cuda')
agent.policy_net.load_state_dict(torch.load('mspacman_pacman.pth'))
agent.epsilon = 0

state, _ = env.reset()
total = 0
done = False
while not done:
    action = agent.select_action(state)
    state, r, term, trunc, _ = env.step(action)
    total += r
    done = term or trunc

print(f'Score test: {total}')
env.close()
