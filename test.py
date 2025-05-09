import torch
import numpy as np
import open3d as o3d
import os
from torch.utils.data import DataLoader
from datasets.custom_dataset import ModelNet40Sphere
from model.vae import PointVAE

# Configurações
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_vae_model.pth"
OUTPUT_DIR = "data/out/true_destruction"
NUM_SAMPLES = 10  # Número de exemplos a serem salvos

def save_pcd(pcd_array, filename):
    """Salva uma nuvem de pontos como arquivo .ply"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    o3d.io.write_point_cloud(filename, pcd)

def main():
    # Criar diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carregar modelo treinado
    model = PointVAE(latent_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Carregar dataset de teste
    test_dataset = ModelNet40Sphere(
        root_dir="data/modelnet40_ply_hdf5_2048",
        file_list="test_files.txt",
        mode='encoder',
        # corruption_rate=0.6
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Processar e salvar exemplos
    for idx, (corrupted, original) in enumerate(test_loader):
        if idx >= NUM_SAMPLES:
            break
            
        # Mover dados para GPU
        corrupted = corrupted.transpose(1, 2).to(DEVICE)  # (1, 3, 2048)
        
        # Inferência
        with torch.no_grad():
            reconstructed, _, _ = model(corrupted)
        
        # Converter para numpy
        corrupted_np = corrupted.cpu().squeeze().transpose(0, 1).numpy()  # (2048, 3)
        original_np = original.squeeze().numpy()  # (2048, 3)
        reconstructed_np = reconstructed.squeeze().cpu().numpy()  # (2048, 3)

        # Salvar arquivos
        save_pcd(original_np, os.path.join(OUTPUT_DIR, f"prefix_pcd_original_{idx}.ply"))
        save_pcd(corrupted_np, os.path.join(OUTPUT_DIR, f"prefix_pcd_corrupted_{idx}.ply"))
        save_pcd(reconstructed_np, os.path.join(OUTPUT_DIR, f"prefix_pcd_reconstructed_{idx}.ply"))

    print(f"Arquivos salvos em {OUTPUT_DIR}!")

if __name__ == "__main__":
    main()