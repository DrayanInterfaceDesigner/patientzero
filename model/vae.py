import torch
import torch.nn as nn

def chamfer_distance(pc1, pc2):
    # pc1: (B, N, 3), pc2: (B, M, 3)
    dist = torch.cdist(pc1, pc2)  # (B, N, M)
    min_dist_pc1_to_pc2, _ = torch.min(dist, dim=2)  # (B, N)
    min_dist_pc2_to_pc1, _ = torch.min(dist, dim=1)  # (B, M)
    chamfer_loss = torch.mean(min_dist_pc1_to_pc2) + torch.mean(min_dist_pc2_to_pc1)
    return chamfer_loss

def chamfer_loss(pc1, pc2):
    # pc1: (B, N, 3), pc2: (B, M, 3)
    dist = torch.cdist(pc1, pc2)          # Distância entre todos os pares
    min_dist_pc1_to_pc2, _ = torch.min(dist, dim=2)
    min_dist_pc2_to_pc1, _ = torch.min(dist, dim=1)
    loss = torch.mean(min_dist_pc1_to_pc2) + torch.mean(min_dist_pc2_to_pc1)
    return loss

class PointVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),          # Camada convolucional 1D
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048*3),      # Saída: 2048 pontos × 3 coordenadas
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.max(x, dim=2)[0]        # Pooling global
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z).view(-1, 3, 2048).transpose(1, 2)  # Formato (B, 2048, 3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
