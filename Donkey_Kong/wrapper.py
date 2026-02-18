import gymnasium as gym
import ale_py
import numpy as np

# Nouveaux chemins pour la version 1.2+
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation  # <--- FrameStack est devenu FrameStackObservation

# Enregistre ALE
gym.register_envs(ale_py)

def make_env(env_name):
    # On lance l'environnement
    env = gym.make(env_name, render_mode="human")
    
    # 1. Conversion en Gris
    env = GrayscaleObservation(env)
    
    # 2. Redimensionnement en 84x84
    env = ResizeObservation(env, (84, 84))
    
    # 3. Empilement de 4 images
    env = FrameStackObservation(env, stack_size=4) # Attention: 'stack_size' au lieu de 'num_stack'
    
    return env

if __name__ == "__main__":
    try:
        env = make_env('ALE/DonkeyKong-v5')
        obs, info = env.reset()
        
        print(f" Succès ! Version Gymnasium: {gym.__version__}")
        print(f" Forme de l'observation : {obs.shape}")
        
        env.step(env.action_space.sample())
        env.close()
    except Exception as e:
        print(f" Erreur persistante : {e}")