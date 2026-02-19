import gymnasium as gym
import numpy as np

class NoWallBump(gym.Wrapper):
    """
    Remplace une action qui mène dans un mur par NOOP (action 0).
    Détecte le mur APRÈS le step réel, en comparant l'obs avant/après.
    """
    def __init__(self, env):
        super().__init__(env)
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Si c'est une direction et que l'image n'a pas changé → mur → on envoie NOOP
        if action in [1, 2, 3, 4] and self._last_obs is not None:
            if self._is_same_frame(obs, self._last_obs):
                # On ne refait pas de step, on signale juste à l'agent que c'était un mur
                info['hit_wall'] = True

        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def _is_same_frame(self, obs1, obs2):
        """Retourne True si les deux frames sont quasi-identiques (mur = pas de mouvement)."""
        arr1 = np.array(obs1, dtype=np.float32)
        arr2 = np.array(obs2, dtype=np.float32)
        return np.mean(np.abs(arr1 - arr2)) < 1.0
