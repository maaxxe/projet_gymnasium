# Projet IA : Ms. Pac-Man — Multi-Agents DQN + Reward Hiérarchisée

Ce projet implémente **5 agents DQN indépendants** (Pac-Man + 4 fantômes) qui apprennent à jouer à Ms. Pac-Man (Atari 2600) uniquement depuis les pixels, sans connaissance des règles. Les récompenses sont hiérarchisées selon l'importance stratégique de chaque événement du jeu.

---

## Structure des fichiers

| Fichier             | Rôle                                                   |
|---------------------|--------------------------------------------------------|
| `dqn_self.py`       | Architecture CNN (cerveau de chaque agent)             |
| `replay_buffer.py`  | Mémoire de replay (une par agent)                      |
| `agent.py`          | Logique de décision et d'apprentissage DQN             |
| `reward_shaping.py` | Hiérarchisation des récompenses par événement          |
| `mspacman_env.py`   | Wrapper ALE + mode manuel clavier                      |
| `run.py`            | Boucle d'entraînement principale (5 agents simultanés) |

---

## 1. `dqn_self.py` — Le Cerveau (CNN)

Définit l'architecture du réseau de neurones convolutif **partagée par tous les agents**. Chaque agent instancie sa propre copie indépendante.

### `DQNCerveau.__init__(self, n_actions)`
- Construit le réseau avec **3 couches convolutives** + **2 couches fully-connected**.
- `n_actions` : nombre d'actions possibles (9 pour Ms. Pac-Man).
- Conv 1 : 4→32 filtres, kernel 8×8, stride 4 — détecte les grandes formes (couloirs, zones).
- Conv 2 : 32→64 filtres, kernel 4×4, stride 2 — détecte les formes moyennes (gommes, fantômes).
- Conv 3 : 64→64 filtres, kernel 3×3, stride 1 — affine les détails fins.
- FC 1 : 64×7×7 → 512 neurones — compresse l'info spatiale en vecteur.
- FC 2 : 512 → n_actions — produit une valeur Q par action possible.

### `DQNCerveau.forward(self, x)`
- Reçoit un état de shape `(batch, 4, 84, 84)` en `uint8`.
- Divise par 255.0 pour normaliser les pixels entre 0 et 1 (stabilise l'apprentissage).
- Passe par les convolutions → aplatit avec `view` → couches FC.
- Retourne un tenseur `(batch, n_actions)` — les valeurs Q de chaque action.

---

## 2. `replay_buffer.py` — La Mémoire

Stocke les expériences passées pour un apprentissage stable. **Chaque agent possède son propre buffer** (50 000 transitions).

### `ReplayBuffer.__init__(self, capacity)`
- Crée une `deque` de taille maximale `capacity`.
- Quand la mémoire est pleine, les souvenirs les plus anciens sont automatiquement supprimés.

### `ReplayBuffer.push(self, state, action, reward, next_state, done)`
- Enregistre une transition complète `(état, action, récompense, état suivant, fin ?)`.
- Appelé à **chaque step** de la boucle pour chacun des 5 agents.

### `ReplayBuffer.sample(self, batch_size)`
- Pioche `batch_size` transitions **au hasard** dans la mémoire.
- Retourne 5 tenseurs PyTorch prêts : `states, actions, rewards, next_states, dones`.
- L'échantillonnage aléatoire **brise les corrélations temporelles** et stabilise le réseau.

### `ReplayBuffer.__len__(self)`
- Retourne le nombre de transitions stockées.
- Utilisé dans `agent.py` pour ne pas entraîner avant d'avoir assez de souvenirs.

---

## 3. `agent.py` — Le Pilote DQN

Contient toute la logique de décision et d'apprentissage. Instancié **5 fois** (1 Pac-Man + 4 fantômes).

### `DQNAgent.__init__(self, n_actions, device, agent_type="pacman")`
- Crée deux réseaux : `policy_net` (apprend en temps réel) et `target_net` (copie figée pour la stabilité).
- `agent_type` : `"pacman"` ou `"ghost"` — les fantômes ont `epsilon_min=0.05` (vs 0.1) car leur rôle antagoniste est plus difficile à apprendre.
- Initialise l'optimiseur **Adam** (lr=0.0001) et un `ReplayBuffer` de 50 000 transitions.
- `epsilon = 1.0` : l'agent joue **100% au hasard** au départ.

### `DQNAgent.select_action(self, state)`
- Stratégie **Epsilon-Greedy** :
  - Avec probabilité `epsilon` → action aléatoire (exploration).
  - Sinon → passe l'état dans `policy_net` et retourne l'action avec la **valeur Q maximale** (exploitation).
- `epsilon` décroît via `epsilon_decay = 0.99995` à chaque entraînement.

### `DQNAgent.train(self, batch_size=32)`
- Ne fait rien si la mémoire contient moins de `batch_size` transitions.
- Pioche un batch avec `self.memory.sample()`.
- Calcule les **valeurs Q actuelles** : `policy_net(states).gather(actions)`.
- Calcule les **valeurs Q cibles** via l'équation de Bellman : `Q = reward + γ × max(target_net(next_states)) × (1 - done)`.
- Calcule la **loss MSE** et effectue une descente de gradient via `optimizer.step()`.
- Réduit `epsilon` après chaque appel.

---

## 4. `reward_shaping.py` — La Hiérarchie des Récompenses ⭐

Transforme la récompense brute ALE en signal pédagogique structuré par importance stratégique.

### Constantes

| Événement         | Reward shapée  | Signal brut ALE        |
|-------------------|:--------------:|------------------------|
| Mort de Pac-Man   | **−50.0**      | perte de vie           |
| Gomme normale     | +1.0           | +10 pts                |
| Power Pellet      | +5.0           | +50 pts                |
| Fruit bonus       | +10.0          | +100 à +500 pts        |
| Manger un fantôme | +20 × combo    | +200/400/800/1600 pts  |
| Finir le niveau   | **+100.0**     | done sans perte de vie |

### `RewardShaper.__init__(self)`
- Initialise les compteurs `prev_score`, `prev_lives`, `prev_dots_left`.

### `RewardShaper.reset(self)`
- Remet tous les compteurs à zéro. Appelé au début de **chaque épisode** dans `run.py`.

### `RewardShaper.shape(self, raw_reward, info, done)`
- **Niveau 1** : détecte la mort via `info["lives"] < prev_lives` → pénalité −50.
- **Niveaux 2–4** : analyse la valeur exacte de `raw_reward` pour identifier l'événement (gomme=10, power pellet=50, fruit=100–500, fantôme=200/400/800/1600 en cascade).
- **Niveau 5** : détecte la fin de niveau si `done=True` sans perte de vie → bonus +100.
- Retourne la `shaped_reward` finale utilisée pour entraîner Pac-Man.

---

## 5. `mspacman_env.py` — L'Environnement

Crée et configure l'environnement ALE avec les bons wrappers dans le bon ordre.

### `make_env(render_mode='human')`
- Désactive le son SDL via `os.environ` avant l'init ALE pour supprimer les messages ALSA.
- Crée `ALE/MsPacman-v5` avec `frameskip=1` (le skip est délégué à `AtariPreprocessing`).
- Applique `AtariPreprocessing` :
  - Grayscale avec `grayscale_newaxis=False` → shape `(84, 84)` (**pas** `(84,84,1)` qui casserait `FrameStack`).
  - Resize 84×84, max-pooling anti-flickering Atari, frame skip 4.
  - `terminal_on_life_loss=False` : la mort est gérée dans `RewardShaper`.
- Applique `FrameStackObservation(stack_size=4)` → shape finale `(4, 84, 84)`.
- `render_mode='human'` : fenêtre visible | `None` : headless (~50× plus rapide).

### `jouer_manuel()`
- Lance le jeu en mode clavier avec `gymnasium.utils.play`.
- Touches : Z (haut), S (bas), Q (gauche), D (droite), zoom ×4.
- Utile pour explorer le jeu manuellement avant l'entraînement.

---

## 6. `run.py` — Le Chef d'Orchestre

Script principal qui instancie et coordonne les 5 agents simultanément.

### Constantes de configuration

| Constante       | Valeur | Rôle                                                    |
|-----------------|--------|---------------------------------------------------------|
| `RENDER`        | True   | Fenêtre visible (False = headless, ~50× plus rapide)    |
| `NUM_EPISODES`  | 1000   | Nombre total de parties jouées                          |
| `BATCH_SIZE`    | 32     | Transitions piochées par entraînement                   |
| `TARGET_UPDATE` | 1000   | Fréquence de sync policy→target (en steps globaux)      |
| `GHOST_NAMES`   | [...]  | Noms des 4 fantômes (logs + fichiers `.pth`)            |

### `charger_modele(agent, path)`
- Vérifie si un fichier `.pth` existe.
- Si oui : charge les poids dans `policy_net` et `target_net`, réduit `epsilon` à 0.2 (l'IA reprend sa progression sans repartir de zéro).
- Si non : démarre un agent vierge (`epsilon = 1.0`).

### Boucle `for episode in range(NUM_EPISODES)`

À chaque épisode, dans l'ordre :
1. `env.reset()` + `shaper.reset()` — remet le jeu et les compteurs à zéro.
2. `pacman_agent.select_action(state)` — Pac-Man choisit son action (epsilon-greedy).
3. `[g.select_action(state) for g in ghost_agents]` — les 4 fantômes choisissent aussi sur le même état visuel.
4. `env.step(pacman_action)` — **seule l'action de Pac-Man** est envoyée à ALE (les fantômes ALE ont leur propre IA interne).
5. `shaper.shape(raw_reward, info, done)` — calcul de la reward shapée pour Pac-Man.
6. Reward fantômes = `−shaped_pacman × 0.5` + bonus +30 si Pac-Man meurt (politique antagoniste).
7. `memory.push(...)` pour chacun des 5 agents.
8. `agent.train(BATCH_SIZE)` pour chacun des 5 agents.
9. `target_net` synchronisé avec `policy_net` tous les `TARGET_UPDATE` steps globaux.
10. `historique_pacman.append(total_reward_pacman)` — enregistre le score de l'épisode.
11. Sauvegarde des 5 fichiers `.pth` tous les 50 épisodes.

### Gestion du Ctrl+C — `KeyboardInterrupt`
- Sauvegarde d'urgence immédiate des 5 modèles `.pth`.
- Affiche un récapitulatif complet des scores :

```
──────────────────────────────────────────────────
   RÉCAPITULATIF — 47 épisodes joués
──────────────────────────────────────────────────
  Score moyen   :    -45.3
  Meilleur score:    320.0  (épisode 31)
  Pire score    :   -200.0  (épisode 2)
  Score final   :     80.0  (dernier épisode)
  Epsilon final : 0.1820
──────────────────────────────────────────────────
```

---

## Logs en temps réel


```
Épisode    0 | Pac-Man:  -137.0 | Blinky: 158.5 | Pinky: 158.5 | Inky: 158.5 | Sue: 158.5 | ε=0.189 | Steps=435
Épisode    1 | Pac-Man:   -30.0 | Blinky: 105.0 | Pinky: 105.0 | Inky: 105.0 | Sue: 105.0 | ε=0.183 | Steps=649
Épisode   50 | Pac-Man:   142.0 | Blinky: -71.0 | ...                                        | ε=0.120 | Steps=4821
```

- **Pac-Man négatif au début** → normal, il apprend encore à éviter les fantômes ✅
- **Fantômes positifs au début** → ils gagnent quand Pac-Man meurt ✅
- **Pac-Man qui monte** → l'IA apprend à manger des gommes et survivre ✅
- **ε qui descend** → l'IA exploite de plus en plus sa stratégie apprise ✅

---

## Fichiers sauvegardés

| Fichier                     | Contenu                    | Fréquence        |
|-----------------------------|----------------------------|------------------|
| `mspacman_pacman.pth`       | Poids du réseau de Pac-Man | Tous les 50 épisodes + Ctrl+C |
| `mspacman_ghost_Blinky.pth` | Poids du réseau de Blinky  | Tous les 50 épisodes + Ctrl+C |
| `mspacman_ghost_Pinky.pth`  | Poids du réseau de Pinky   | Tous les 50 épisodes + Ctrl+C |
| `mspacman_ghost_Inky.pth`   | Poids du réseau de Inky    | Tous les 50 épisodes + Ctrl+C |
| `mspacman_ghost_Sue.pth`    | Poids du réseau de Sue     | Tous les 50 épisodes + Ctrl+C |

---

## Paramètres et justifications

| Paramètre       | Valeur  | Justification                                                              |
|-----------------|---------|----------------------------------------------------------------------------|
| Learning Rate   | 0.0001  | Convergence stable, évite d'oublier les succès passés                      |
| Gamma (γ)       | 0.99    | IA patiente : comprend que manger une power pellet prépare la chasse       |
| Epsilon Decay   | 0.99995 | Décroissance lente : les niveaux sont complexes, l'exploration est longue  |
| Batch Size      | 32      | Compromis VRAM/stabilité optimal sur GPU RTX                               |
| Target Update   | 1000    | La cible ne bouge pas trop vite, évite la divergence du réseau             |
| Buffer Size     | 50 000  | Par agent — ~250 000 transitions totales en mémoire simultanément          |

---

## Installation et lancement

```bash
pip install gymnasium[atari] ale-py torch
python run.py
```

Pour entraîner sans fenêtre (mode rapide ~50× plus vite) :
```python
# Dans run.py, ligne 13 :
RENDER = False
```

Pour un entrainement optimal ( pour que ça tourne même si le terminal est fermée):
```python 
# lancer 
nohup python run.py & 
# avec RENDER = False,  
```
