# reward_shaping.py
# Hiérarchie des récompenses pour Ms. Pac-Man
# =============================================
# Niveau 1 (vital)     : survivre, ne pas mourir
# Niveau 2 (tactique)  : manger les gommes (dots)
# Niveau 3 (stratégique): manger les power pellets
# Niveau 4 (bonus)     : manger les fantômes sous effet power pellet
# Niveau 5 (objectif)  : finir le niveau (toutes les gommes mangées)

REWARD_MORT           = -50.0   # Pénalité forte pour la mort
REWARD_DOT            =   1.0   # Gomme normale (+10 pts dans le jeu)
REWARD_POWER_PELLET   =   5.0   # Power pellet (+50 pts, ouvre le mode chasse)
REWARD_FANTOME        =  20.0   # Manger un fantôme (200-1600 pts en cascade)
REWARD_FRUIT          =  10.0   # Fruit bonus (cerise, fraise, etc.)
REWARD_NIVEAU_FINI    = 100.0   # Finir le niveau = objectif ultime

class RewardShaper:
    """
    Transforme la reward brute ALE en reward hiérarchisée
    en comparant l'état RAM avant/après chaque step.
    """
    def __init__(self):
        self.prev_score      = 0
        self.prev_lives      = 3 # ALE/MsPacman-v5 commence avec 3 vies
        self.prev_dots_left  = None  # On initialise au 1er step

    def reset(self):
        self.prev_score     = 0
        self.prev_lives     = 3
        self.prev_dots_left = None

    def shape(self, raw_reward, info, done):
        """
        Calcule la reward shapée à partir de la reward brute ALE
        et des infos de l'environnement.

        Paramètres
        ----------
        raw_reward  : float  — reward brute retournée par env.step()
        info        : dict   — dictionnaire retourné par env.step()
        done        : bool   — True si l'épisode est terminé

        Retour
        ------
        shaped_reward : float
        """
        shaped = 0.0
        current_lives = info.get("lives", self.prev_lives)

        # --- Niveau 1 : Mort ---
        if current_lives < self.prev_lives:
            shaped += REWARD_MORT
            self.prev_lives = current_lives

        # --- Niveaux 2-4 : Décomposition de la reward brute ---
        # La reward brute ALE encode déjà le score du jeu.
        # On amplifie différemment selon le palier de points gagné.
        if raw_reward > 0:
            # Gomme normale : ~10 pts dans le jeu
            if raw_reward <= 10:
                shaped += REWARD_DOT * (raw_reward / 10.0)

            # Power pellet : 50 pts
            elif raw_reward == 50:
                shaped += REWARD_POWER_PELLET

            # Fruit bonus : 100-5000 pts selon le niveau
            elif 100 <= raw_reward <= 500:
                shaped += REWARD_FRUIT

            # Manger un fantôme : 200, 400, 800, 1600 pts (en cascade)
            elif raw_reward in (200, 400, 800, 1600):
                # Bonus supplémentaire si combo (cascade de fantômes)
                combo_bonus = raw_reward / 200.0  # 1x, 2x, 4x, 8x
                shaped += REWARD_FANTOME * combo_bonus

            # Autre reward (score divers)
            else:
                shaped += raw_reward / 100.0

        # --- Niveau 5 : Finir le niveau ---
        # Détecté si done=True SANS perte de vie (toutes les gommes mangées)
        if done and current_lives == self.prev_lives and current_lives > 0:
            shaped += REWARD_NIVEAU_FINI

        # --- Bonus de survie : encourage Pac-Man à rester en vie ---
        # Contre-balance la pénalité de mort et pousse à explorer
        if current_lives == self.prev_lives:
            shaped += 0.1

        return shaped
