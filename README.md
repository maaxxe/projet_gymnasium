# projet_gymnasium

## Le chemin "Rapide" (Utiliser une librairie)

Pour obtenir des résultats rapidement (comme les 5100 points dont tu parlais), je te conseille d'utiliser Stable Baselines3. C'est une bibliothèque qui contient des implémentations de DQN déjà optimisées.
Bash

pip install stable-baselines3 shimmy

Python

from stable_baselines3 import DQN
import gymnasium as gym

# Création de l'environnement
env = gym.make("ALE/DonkeyKong-v5", render_mode="human")

# Initialisation du modèle DQN
model = DQN("CnnPolicy", env, verbose=1)

# Lancer l'entraînement
model.learn(total_timesteps=10000)