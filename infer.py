import torch
import numpy as np
import os
from model.vae import VAE
from datasets.custom_dataset import ModelNet40VAE
import open3d as o3d  # Para salvar as nuvens

# Configurações
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "data/out"
os.makedirs(OUT_DIR, exist_ok=True)

# Carregar modelo
model = VAE(latent_dim=128).to(DEVICE)
model.load_state_dict(torch.load('vae_epoch_49.pth'))
model.eval()

# Carregar dataset de teste
h5_path = "data/modelnet40_ply_hdf5_2048/ply_data_test0.h5"
test_dataset = ModelNet40VAE(h5_path, mode='encoder')

# Processar e salvar exemplos
for i in range(3):  # Salvar 3 exemplos
    corrupted_pc, original_pc = test_dataset[i]
    corrupted_pc = corrupted_pc.to(DEVICE)
    
    with torch.no_grad():
        reconstructed, _, _ = model(corrupted_pc.unsqueeze(0))
    
    # Converter para numpy e remodelar
    corrupted_np = corrupted_pc.cpu().numpy().reshape(-1, 3)
    reconstructed_np = reconstructed.squeeze().cpu().numpy().reshape(-1, 3)
    original_np = original_pc.numpy().reshape(-1, 3)
    
    # Salvar como .ply
    def save_pcd(arr, filename):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr)
        o3d.io.write_point_cloud(filename, pcd)
    
    save_pcd(corrupted_np, os.path.join(OUT_DIR, f'corrupted_{i}.ply'))
    save_pcd(reconstructed_np, os.path.join(OUT_DIR, f'reconstructed_{i}.ply'))
    save_pcd(original_np, os.path.join(OUT_DIR, f'original_{i}.ply'))

print("PCDs salvas em data/out/")