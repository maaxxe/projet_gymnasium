import gymnasium as gym
import ale_py
import numpy as np
import os

gym.register_envs(ale_py)

#  Supprime les messages ALSA underrun
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["AUDIODEV"] = "null"

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
        repeat_action_probability=0.0
        
    )

    # AtariPreprocessing fait tout en une fois :
    # - Grayscale (sans keep_dim → shape (84,84))
    # - Resize 84x84
    # - Max pooling sur 2 frames (évite le flickering Atari)
    # - Frame skip (4 frames par action)
    # - Normalisation optionnelle désactivée (on le fait dans le CNN)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,   # On gère la mort dans RewardShaper
        grayscale_obs=True,
        grayscale_newaxis=False,       # ← CLÉ : shape (84,84) PAS (84,84,1)
        scale_obs=False                # Pas de normalisation ici (fait dans CNN /255)
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
