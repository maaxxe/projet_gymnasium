import torch
from mspacman_env import make_env  # Ton wrapper custom
from agent import DQNAgent
import os

env = make_env()
agent = DQNAgent(env.action_space.n, 'cuda')



# device automatique (crash si pas de GPU avec 'cuda' )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_env(render_mode='human')
agent = DQNAgent(env.action_space.n, device)

#  vérification de l'existence du modèle avant torch.load
if not os.path.exists("mspacman_pacman.pth"):
    print("Aucun modèle trouvé. Lance d'abord run.py.")
    env.close()
    exit()


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
