import gymnasium as gym
from gymnasium import spaces

class NoWallBump(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Actions MsPacman: 0=NOOP,1=UP,2=RIGHT,3=LEFT,4=DOWN,5=START,etc
        self.wall_actions = [1,2,3,4]  # Directions
        
    def action(self, action):
        if action in self.wall_actions:
            obs, _, _, _, info = self.env.step(action)
            if self.is_wall(obs):  # Contre mur ?
                return 0  # NOOP au lieu
        return action
    
    def is_wall(self, obs):
        # Détecte mur = bord image noir (84x84 grayscale)
        gray = obs.mean(axis=-1) if obs.ndim == 3 else obs
        wall_threshold = 20  # Pixel très noir = mur
        return (gray[0:5,:].mean() < wall_threshold or 
                gray[-5:,:].mean() < wall_threshold or
                gray[:,0:5].mean() < wall_threshold or
                gray[:,-5:].mean() < wall_threshold)
