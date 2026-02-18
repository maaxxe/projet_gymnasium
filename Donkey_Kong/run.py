import gymnasium as gym
import torch
import numpy as np
from wrapper import make_env       # 'wrapper' en minuscule comme ton fichier
from dqn_self import DQNCerveau    # Ton fichier s'appelle 'dqn_self.py'
from agent import DQNAgent         # Import de l'agent
from replay_buffer import ReplayBuffer # Import du buffer
import os


#onfiguration du CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" L'entraînement utilise : {device}")




# 2. Initialisation
env = make_env('ALE/DonkeyKong-v5')
n_actions = env.action_space.n
agent = DQNAgent(n_actions, device)



if os.path.exists("donkey_kong_dqn.pth"):
    print(" Ancien modèle trouvé ! Chargement des neurones...")
    agent.policy_net.load_state_dict(torch.load("donkey_kong_dqn.pth"))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.epsilon = 0.2  # On réduit le hasard (l'IA sait déjà un peu jouer)
else:
    print(" Aucun modèle trouvé. L'IA part de zéro.")

# On remplace la mémoire vide par celle de l'agent
agent.memory = ReplayBuffer(50000) 

BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE = 1000 # On met à jour le cerveau cible tous les 1000 pas
num_episodes = 500

# 3. Boucle principale d'entraînement
for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0
    
    for t in range(10000): # Max pas par partie
        # L'agent choisit une action
        action = agent.select_action(state)
        
        # Le jeu s'exécute
        next_state, reward, terminated, truncated, info = env.step(action) #ajout reward si monte ou saute au dessus d un tonneau
        done = terminated or truncated
        
        # On enregistre dans la mémoire
        agent.memory.push(state, action, reward, next_state, done)
        
        # On entraîne le cerveau
        agent.train(BATCH_SIZE)
        
        state = next_state
        total_reward += reward
        
        # Mise à jour périodique du cerveau cible
        if t % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        if done:
            break
            
    print(f"Épisode {episode} | Score: {total_reward} | Epsilon: {agent.epsilon:.2f}")

    # Optionnel : Sauvegarder le modèle toutes les 50 parties
    if episode % 50 == 0:
        torch.save(agent.policy_net.state_dict(), "donkey_kong_dqn.pth")

env.close()