Projet IA : Donkey Kong Reinforcement Learning

#Ce projet implémente un agent de Deep Q-Learning (DQN) capable d'apprendre à jouer à Donkey Kong sur Atari 2600. L'IA n'a aucune connaissance des règles : elle apprend uniquement en observant les pixels et en recevant des récompenses (score).
## Description des Classes

# 1. wrapper.py (L'interface visuelle)

Cette classe transforme la sortie brute du jeu en données optimisées pour le réseau de neurones.

    GrayscaleObservation : Réduit la complexité en passant de la couleur au noir et blanc.

    ResizeObservation : Redimensionne l'image à 84x84, éliminant les détails inutiles pour accélérer le calcul.

    FrameStackObservation : Empile les 4 dernières images. C'est ce qui permet à l'IA de "voir" si un tonneau avance ou si Mario saute (perception du mouvement).

# 2. dqn_self.py (Le Cerveau - CNN)

Définit l'architecture du Réseau de Neurones Convolutif.

    Couches Convolutives : Agissent comme des filtres qui repèrent les formes (échelles, plateformes, ennemis).

    Couches Linéaires : Traduisent ces formes en probabilités d'actions (monter, sauter, aller à droite).

    Normalisation : Divise les valeurs de pixels par 255 pour stabiliser l'apprentissage (valeurs entre 0 et 1).

# 3. replay_buffer.py (La Mémoire)

Gère l'expérience passée de l'agent.

    Stockage : Garde en mémoire les transitions (état, action, récompense, état suivant).

    Échantillonnage (Sampling) : Permet à l'IA de piocher des souvenirs au hasard pour s'entraîner. Cela évite que l'IA ne se focalise uniquement sur ce qu'elle vient de vivre, stabilisant ainsi son cerveau.

# 4. agent.py (Le Pilote)

Contient la logique de décision et d'évolution.

    Epsilon-Greedy : Gère l'équilibre entre exploration (tenter de nouvelles choses au hasard) et exploitation (utiliser ce qu'elle a appris pour gagner).

    Double DQN : Utilise deux réseaux (policy_net et target_net) pour éviter que l'IA ne surestime la valeur de certaines actions, ce qui rend l'apprentissage beaucoup plus robuste.

# 5. run.py (Le Chef d'Orchestre)

C'est le script principal qui lie tous les modules.

    Initialise l'environnement sur CUDA (GPU NVIDIA) pour une vitesse maximale.

    Gère la boucle d'entraînement par épisodes.

    Sauvegarde/Chargement : Utilise le fichier donkey_kong_dqn.pth pour que l'IA conserve ses acquis d'une session à l'autre.

#  Suivi de l'Apprentissage

Les résultats sont visibles directement dans le terminal :

    Score : La somme des récompenses obtenues pendant une partie.

    Epsilon : Le taux de hasard. S'il est à 0.10, l'IA joue sérieusement 90% du temps.

#  Sauvegarde et Reprise

Le script est configuré pour :

    Sauvegarder automatiquement le modèle tous les 50 épisodes.

    Charger automatiquement donkey_kong_dqn.pth s'il est présent au lancement, permettant une progression continue sur plusieurs jours.




### print 

Épisode (L'expérience) : C'est le nombre de parties jouées depuis le début.

Score (La performance) : C'est le total des points gagnés dans la partie.

Epsilon (Le hasard vs l'Intelligence) : C'est le paramètre de la stratégie Epsilon-Greedy (le pourcentage de hasard).


### parmetres

1. Learning Rate (Taux d'apprentissage) : 0.0001

    C'est la vitesse à laquelle l'IA modifie ses neurones lorsqu'elle fait une erreur.

    Justification : Un taux trop élevé (ex: 0.01) ferait "paniquer" l'IA, lui faisant oublier ses anciens succès à chaque nouvelle erreur. 0.0001 est une valeur prudente qui permet une convergence stable vers une solution optimale.

2. Gamma (γ) : 0.99

    C'est le facteur de remise (discount factor). Il détermine l'importance des récompenses futures par rapport aux récompenses immédiates.

    Justification : Avec 0.99, l'IA est "patiente". Elle comprend que monter une échelle (pas de points immédiats) est nécessaire pour atteindre le haut et gagner beaucoup de points plus tard. Si γ était proche de 0, l'IA serait "myope" et ne chercherait que les points instantanés.

3. Epsilon Decay : 0.9999 ou 0.99995

    C'est la vitesse à laquelle l'IA arrête de jouer au hasard pour commencer à utiliser sa propre stratégie.

    Justification : Dans Donkey Kong, les récompenses sont rares (il faut monter longtemps avant de marquer). On utilise une décroissance très lente pour forcer l'IA à explorer le niveau pendant des milliers d'étapes avant de se fixer sur une stratégie définitive.

4. Batch Size : 32

    C'est le nombre de souvenirs que l'IA pioche dans sa mémoire à chaque étape d'entraînement.

    Justification : 32 est le compromis idéal entre vitesse de calcul sur ta carte RTX et stabilité statistique. Cela permet de faire des mises à jour régulières sans saturer la mémoire vidéo (VRAM).

5. Target Update : 1000

    C'est la fréquence à laquelle on synchronise le "cerveau cible" avec le "cerveau actif".

    Justification : Si on le faisait à chaque pas, l'IA essaierait de prédire une cible qui change en même temps qu'elle apprend (comme courir après un mirage). En attendant 1000 pas, on stabilise l'objectif, ce qui empêche le réseau de diverger.