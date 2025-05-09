import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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

# class PointVAE(nn.Module):
#     def __init__(self, latent_dim=256):
#         super().__init__()
        
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv1d(3, 64, 1),          # Camada convolucional 1D
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv1d(64, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv1d(128, 256, 1),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.2),
#         )
        
#         self.fc_mu = nn.Linear(256, latent_dim)
#         self.fc_logvar = nn.Linear(256, latent_dim)
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, 2048*3),      # Saída: 2048 pontos × 3 coordenadas
#         )

#     def encode(self, x):
#         x = self.encoder(x)
#         x = torch.max(x, dim=2)[0]        # Pooling global
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, logvar

#     def decode(self, z):
#         return self.decoder(z).view(-1, 3, 2048).transpose(1, 2)  # Formato (B, 2048, 3)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decode(z)
#         return recon, mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
import torch
import torch.nn as nn
import torch.nn.functional as F

class StdPool(nn.Module):
    def forward(self, x):
        return torch.std(x, dim=2, keepdim=True)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch)
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x):
        return F.gelu(self.conv(x) + self.shortcut(x))

class ResMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim),
        )
        self.norm = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        return self.norm(x + self.block(x))

class PointVAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            ResidualConvBlock(3, 128),
            ResidualConvBlock(128, 256),
            ResidualConvBlock(256, 512),
            ResidualConvBlock(512, 1024)
        )
        
        # Pooling Multinível
        self.pool = nn.ModuleDict({
            'max': nn.AdaptiveMaxPool1d(1),
            'avg': nn.AdaptiveAvgPool1d(1),
            'std': StdPool()
        })
        
        # Projeção Latente
        self.fc_mu = nn.Linear(3072, latent_dim)  # 1024*3
        self.fc_logvar = nn.Linear(3072, latent_dim)
        
        # Decoder
        self.decoder = self.build_decoder(latent_dim)
        self.register_buffer('grid', self.create_grid(64))
    
    def create_grid(self, size):
        x = np.linspace(-0.5, 0.5, num=size)
        y = np.linspace(-0.5, 0.5, num=size)
        grid = np.stack(np.meshgrid(x,y), -1).reshape(-1,2)
        return torch.FloatTensor(grid)
    
    def build_decoder(self, latent_dim):
        return nn.Sequential(
            nn.Linear(latent_dim + 2, 1024),
            ResMLPBlock(1024),
            ResMLPBlock(1024),
            nn.Linear(1024, 512),
            ResMLPBlock(512),
            nn.Linear(512, 3)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        max_pool = self.pool['max'](x).flatten(1)
        avg_pool = self.pool['avg'](x).flatten(1)
        std_pool = self.pool['std'](x).flatten(1)
        features = torch.cat([max_pool, avg_pool, std_pool], dim=1)
        return self.fc_mu(features), self.fc_logvar(features)
    
    def decode(self, z):
        B = z.size(0)
        grid = self.grid.repeat(B, 1, 1)
        z_exp = z.unsqueeze(1).repeat(1, grid.size(1), 1)
        x = torch.cat([z_exp, grid], dim=-1).view(-1, z.size(-1)+2)
        
        for layer in self.decoder:
            x = layer(x)
            
        return x.view(B, -1, 3)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

# def hybrid_loss(recon, target, mu, logvar, beta=0.05):
#     # Chamfer Distance
#     dist = torch.cdist(recon, target)
#     min_dist_1 = torch.min(dist, dim=2)[0].mean()
#     min_dist_2 = torch.min(dist, dim=1)[0].mean()
#     cd_loss = min_dist_1 + min_dist_2
    
#     # KL Divergence
#     kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
#     return cd_loss + beta * kl_loss