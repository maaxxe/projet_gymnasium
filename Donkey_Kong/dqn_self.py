import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNCerveau(nn.Module):
    def __init__(self, n_actions):
        super(DQNCerveau, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Avec 3 couches, la sortie est 7x7. Avec tes 2 couches, c'était 9x9.
        # On utilise 7x7 pour être fidèle au papier de recherche original.
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # Important : Diviser par 255 pour que les pixels soient entre 0 et 1
        x = x.float() / 255.0 
        x = self.conv(x)
        x = x.view(x.size(0), -1) 
        return self.fc(x)