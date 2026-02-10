import gymnasium as gym
import ale_py

# Cette ligne lie ALE à Gymnasium
gym.register_envs(ale_py)

# Création de l'environnement (v5 est la version standard actuelle)
env = gym.make('ALE/Breakout-v5', render_mode='human')

obs, info = env.reset()

# On fait une boucle pour tester si le moteur tourne
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()