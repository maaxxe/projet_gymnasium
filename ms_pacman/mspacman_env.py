import gymnasium as gym
import ale_py
import numpy as np
import os
from wall_avoid_wrapper import NoWallBump

# Tentative d'importation robuste des wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

#  Supprime les messages ALSA underrun
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["AUDIODEV"] = "null"

class LimitActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # On définit les actions que l'on veut garder :
        # 0: NOOP, 1: UP, 2: RIGHT, 3: LEFT, 4: DOWN
        self.mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        self.action_space = gym.spaces.Discrete(5)

    def action(self, action):
        return self.mapping.get(action, 0)

def make_env(render_mode='human'): # humain par default change dans run.py
    """
    Wrapper Ms. Pac-Man compatible avec toutes versions Gymnasium.
    Produit un état de forme (4, 84, 84) — identique au projet DonkeyKong.
    
    Ordre CRITIQUE des wrappers :
    1. AtariPreprocessing : grayscale + resize 84x84 + skip 4 frames
       → sortie shape (84, 84), dtype uint8
    2. FrameStackObservation : empile 4 frames
       → sortie shape (4, 84, 84)
    
    On N'utilise PAS GrayscaleObservation + keep_dim=True car cela produit
    (1,84,84) puis FrameStack le transforme en (4,1,84,84) → CRASH.
    """
    env = gym.make(
        'ALE/MsPacman-v5',
        render_mode=render_mode,
        frameskip=1,          # AtariPreprocessing gère le frameskip lui-même
        full_action_space=False,  # ← réduit à 5 actions : NOOP, UP, RIGHT, LEFT, DOWN
        repeat_action_probability=0.0,
    )

   
    env = LimitActions(env) # limite les actions a 5


    

    # ATTENTION : On met frame_skip=1 ici car tu as ajouté FrameSkip(skip=4) manuellement.
    env = AtariPreprocessing(
        env,
        noop_max=30,           
        frame_skip=4,         # On laisse AtariPreprocessing gérer le saut de 4 (plus stable)
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False
    )

    # FrameStackObservation : empile 4 frames → (4, 84, 84)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    return env


def jouer_manuel():
    """Mode manuel clavier pour explorer le jeu."""
    from gymnasium.utils.play import play
    env_manual = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
    play(env_manual, keys_to_action={
        (ord("z"),): 1,  # Haut
        (ord("s"),): 4,  # Bas
        (ord("q"),): 3,  # Gauche
        (ord("d"),): 2,  # Droite
    }, fps=30, zoom=4)
    env_manual.close()


if __name__ == "__main__":
    # Test rapide pour vérifier la forme de l'état
    env = make_env()
    state, info = env.reset()
    print(f"Shape de l'état : {state.shape}")   # Attendu : (4, 84, 84)
    print(f" dtype : {state.dtype}")              # Attendu : uint8
    print(f" Actions disponibles : {env.action_space.n}")  # Attendu : 9
    env.close()
