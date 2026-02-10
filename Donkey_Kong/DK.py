import gymnasium as gym
import ale_py
from gymnasium.utils.play import play

# Cette ligne lie ALE à Gymnasium
gym.register_envs(ale_py)

# Création de l'environnement (v5 est la version standard actuelle)
env = gym.make('ALE/DonkeyKong-v5', render_mode='human')

obs, info = env.reset()

def jouer():
    print("🎮 Mode Manuel activé !")
    print("ASTUCE : Appuie sur ESPACE pour démarrer si l'image est fixe.")
    
    env_manual = gym.make('ALE/DonkeyKong-v5', render_mode='rgb_array')
    
    # On ajoute zoom=4 pour agrandir la fenêtre
    play(env_manual, keys_to_action={
        (ord("z"),): 2,      # Haut
        (ord("s"),): 5,      # Bas
        (ord("q"),): 4,      # Gauche
        (ord("d"),): 3,      # Droite
        (ord(" "),): 1,      # Saut (Espace)
    }, fps=30, zoom=4) # <--- ZOOM AJOUTÉ ICI
    
    env_manual.close()

def test():
    # On fait une boucle pour tester si le moteur tourne
    for _ in range(100):
        # Action aléatoire (remplacez par votre IA plus tard)
        action = env.action_space.sample()
        
        # Exécution de l'action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Si le jeu est fini ou bloqué, on recommence
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    jouer()