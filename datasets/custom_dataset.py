import h5py
import numpy as np
import torch
import os
from torch.utils.data import Dataset

def normalize_to_sphere(pc):
    """
    Normaliza a nuvem de pontos para uma esfera unitária:
    1. Centraliza no centroide.
    2. Escala para o ponto mais distante ter norma 1.
    """
    centroid = np.mean(pc, axis=0)
    pc_centered = pc - centroid
    max_distance = np.max(np.linalg.norm(pc_centered, axis=1))
    if max_distance == 0:
        return pc_centered
    pc_normalized = pc_centered / max_distance
    return pc_normalized

class ModelNet40Sphere(Dataset):
    def __init__(self, root_dir, file_list, mode='train', corruption_rate=0.5, num_points=2048):
        self.mode = mode
        self.corruption_rate = corruption_rate
        self.num_points = num_points
        self.data = []
        
        with open(os.path.join(root_dir, file_list), 'r') as f:
            h5_files = [os.path.join(root_dir, line.strip()) for line in f]
            
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                pcds = f['data'][:].astype('float32')  # (N, 2048, 3)
                for pcd in pcds:
                    pcd_normalized = normalize_to_sphere(pcd) 
                    self.data.append(pcd_normalized)
        
    def __len__(self):
        return len(self.data)
    
    def corrupt_pointcloud(self, pc):
        return pc
    
    def __getitem__(self, idx):
        original_pc = self.data[idx]  # (2048, 3), já normalizado
        
        if self.mode == 'encoder':
            corrupted_pc = self.corrupt_pointcloud(original_pc)
            
            padded_pc = np.zeros((self.num_points, 3))
            padded_pc[:corrupted_pc.shape[0]] = corrupted_pc
            return torch.tensor(padded_pc).float(), torch.tensor(original_pc).float()
        else:
            return torch.tensor(original_pc).float()