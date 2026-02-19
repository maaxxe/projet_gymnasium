# run.py — Multi-agents Ms. Pac-Man avec reward hiérarchisée
import torch
import numpy as np
import os
from mspacman_env import make_env
from agent import DQNAgent
from reward_shaping import RewardShaper



# === Progression totale =======================================
episodes_precedents = 0
if os.path.exists("progression.txt"):
    with open("progression.txt", "r") as f:
        episodes_precedents = int(f.read())
    print(f"    Progression totale : {episodes_precedents} épisodes joués au total")
else:
    print(f"    Premier démarrage — aucun historique trouvé")


# === Configuration ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entraînement sur : {device}")

RENDER = False   # ← Mettre False pour entraîner sans fenêtre (plus rapide) de ~3000 steps/s a 60 steps/s
NUM_EPISODES = 1000 # Nombre d'épisodes d'entraînement (ajuster selon le temps disponible)
BATCH_SIZE    = 64      # Taille du batch pour l'entraînement DQN
TARGET_UPDATE = 1000    # Nombre de steps entre chaque mise à jour du Target Network
GHOST_NAMES   = ["Blinky", "Pinky", "Inky", "Sue"]  

# === Environnement ============================================
env = make_env(render_mode='human' if RENDER else None)   # ← ICI, remplace l'ancien make_env()
n_actions = env.action_space.n

# === Agents ===================================================
pacman_agent = DQNAgent(n_actions, device, agent_type="pacman")
ghost_agents  = [DQNAgent(n_actions, device, agent_type="ghost")
                 for _ in GHOST_NAMES]

# Chargement des modèles sauvegardés si disponibles
def charger_modele(agent, path):
    if os.path.exists(path):
        agent.policy_net.load_state_dict(torch.load(path, map_location=device))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        # FIXÉ : on recharge aussi l'epsilon sauvegardé si disponible
        eps_path = path.replace(".pth", "_epsilon.txt")
        if os.path.exists(eps_path):
            with open(eps_path) as f:
                agent.epsilon = float(f.read())
        else:
            agent.epsilon = 0.5
        print(f"  Modèle chargé : {path} (ε={agent.epsilon:.3f})")
    else:
        print(f"  Nouveau modèle : {path}")

def sauvegarder(episode_total):
    torch.save(pacman_agent.policy_net.state_dict(), "mspacman_pacman.pth")
    # FIXÉ : sauvegarde de l'epsilon pour reprendre correctement
    with open("mspacman_pacman_epsilon.txt", "w") as f:
        f.write(str(pacman_agent.epsilon))
    for i, g_agent in enumerate(ghost_agents):
        path = f"mspacman_ghost_{GHOST_NAMES[i]}.pth"
        torch.save(g_agent.policy_net.state_dict(), path)
        with open(path.replace(".pth", "_epsilon.txt"), "w") as f:
            f.write(str(g_agent.epsilon))
    with open("progression.txt", "w") as f:
        f.write(str(episode_total))
    print(f"  Modèles sauvegardés (total : {episode_total} épisodes)")

charger_modele(pacman_agent, "mspacman_pacman.pth")
for i, g_agent in enumerate(ghost_agents):
    charger_modele(g_agent, f"mspacman_ghost_{GHOST_NAMES[i]}.pth")

# === Reward Shaper ============================================
shaper = RewardShaper()

# === Compteur global de steps (pour TARGET_UPDATE) ============
global_step = 0

# === Boucle principale ========================================

historique_pacman = []  # Pour suivre les scores de Pac-Man
try:
    for episode in range(NUM_EPISODES):
        state, info = env.reset()
        shaper.reset()
        

        total_reward_pacman = 0.0
        total_reward_ghosts = [0.0] * 4

        for t in range(15000):
            global_step += 1

            # == Pac-Man choisit son action ==
            pacman_action = pacman_agent.select_action(state)

            # == Fantômes choisissent leur action (même obs visuelle) ==
            # Note : Dans ALE, seule l'action de Pac-Man est envoyée à l'env.
            # Les fantômes ont une IA interne ALE. Ici, on entraîne des cerveaux
            # "fantômes" en leur donnant le même état pixel et une reward inversée,
            # simulant leur rôle antagoniste.
            ghost_actions = [g.select_action(state) for g in ghost_agents]

            # == Step de l'environnement (action de Pac-Man uniquement) ==
            next_state, raw_reward, terminated, truncated, info = env.step(pacman_action)
            done = terminated or truncated


            # mise à jour de power_pellet_active avant le shaping
            if raw_reward == 50:
                pacman_agent.set_power_pellet(True)
            elif done or info.get("lives", 3) < shaper.prev_lives:
                pacman_agent.set_power_pellet(False)

            # == Reward hiérarchisée pour Pac-Man ==
            shaped_reward_pacman = shaper.shape(raw_reward, info, done)

            # == Reward inversée pour les fantômes ==
            # Les fantômes sont récompensés quand Pac-Man subit une pénalité
            shaped_reward_ghost = -shaped_reward_pacman * 0.5
            # Bonus fantôme si Pac-Man meurt
            if shaped_reward_pacman <= -50:
                shaped_reward_ghost += 30.0

            # == Enregistrement mémoire ==
            pacman_agent.memory.push(state, pacman_action, shaped_reward_pacman, next_state, done)
            for i, g_agent in enumerate(ghost_agents):
                g_agent.memory.push(state, ghost_actions[i], shaped_reward_ghost, next_state, done)

            # == Entraînement ==
            pacman_agent.train(BATCH_SIZE)
            for g_agent in ghost_agents:
                g_agent.train(BATCH_SIZE)

            # == Mise à jour Target Networks ==
            if global_step % TARGET_UPDATE == 0:
                pacman_agent.target_net.load_state_dict(pacman_agent.policy_net.state_dict())
                for g_agent in ghost_agents:
                    g_agent.target_net.load_state_dict(g_agent.policy_net.state_dict())

            state = next_state
            total_reward_pacman += shaped_reward_pacman
            for i in range(4):
                total_reward_ghosts[i] += shaped_reward_ghost

            if done:
                break

        # == Logs ==
        ghost_scores = " | ".join(
            f"{GHOST_NAMES[i]}: {total_reward_ghosts[i]:.1f}" for i in range(4)
        )
        print(
            f"Épisode {episode:4d} | "
            f"Pac-Man: {total_reward_pacman:7.1f} | "
            f"{ghost_scores} | "
            f"ε={pacman_agent.epsilon:.3f} | "
            f"Steps={t}"
        )
        historique_pacman.append(total_reward_pacman)

        # == Sauvegarde toutes les 50 parties ==
        if episode % 50 == 0:
            sauvegarder(episodes_precedents + episode)

except KeyboardInterrupt:
    print("\n  Interruption — sauvegarde d'urgence...")
    sauvegarder(episodes_precedents + episode)
finally:
     # == Récapitulatif des scores ==
    if historique_pacman:
        nb = len(historique_pacman)
        print(f"\n{'='*50}")
        print(f"   RÉCAPITULATIF — {nb} épisodes joués")
        print(f"{'='*50}")
        print(f"  Score moyen   : {sum(historique_pacman)/nb:8.1f}")
        print(f"  Meilleur score: {max(historique_pacman):8.1f}  (épisode {historique_pacman.index(max(historique_pacman))})")
        print(f"  Pire score    : {min(historique_pacman):8.1f}  (épisode {historique_pacman.index(min(historique_pacman))})")
        print(f"  Score final   : {historique_pacman[-1]:8.1f}  (dernier épisode)")
        print(f"  Epsilon final : {pacman_agent.epsilon:.4f}")
        print(f"{'='*50}\n")



    env.close()


