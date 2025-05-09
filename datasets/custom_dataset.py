import h5py
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from scipy.spatial import cKDTree

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
        
        # Carregar todos os arquivos HDF5 listados no file_list
        with open(os.path.join(root_dir, file_list), 'r') as f:
            h5_files = [os.path.join(root_dir, line.strip()) for line in f]
            
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                pcds = f['data'][:].astype('float32')  # (N, 2048, 3)
                for pcd in pcds:
                    pcd_normalized = normalize_to_sphere(pcd)  # Normaliza para esfera
                    self.data.append(pcd_normalized)
        
        # Diretório para salvar os dados corrompidos
        self.corrupted_dir = os.path.join(root_dir, "corrupted_dataset")
        os.makedirs(self.corrupted_dir, exist_ok=True)
        
        # Salvar dados corrompidos se estiver no modo 'encoder'
        if self.mode == 'encoder':
            self.save_corrupted_data()
    
    def __len__(self):
        return len(self.data)
    
    def corrupt_pointcloud(self, pc, idx):
        """Corrompe a nuvem de pontos removendo clusters aleatórios de forma mais agressiva."""
        seed = idx  # Semente baseada no índice para reprodutibilidade
        rng = np.random.default_rng(seed)
        
        num_points = pc.shape[0]
        num_to_remove = int(self.corruption_rate * num_points)
        
        if num_to_remove == 0:
            return pc
        
        # Cria KDTree para busca eficiente de vizinhos
        tree = cKDTree(pc)
        
        # Máscara para pontos mantidos (inicialmente todos)
        mask = np.ones(num_points, dtype=bool)
        available_indices = np.arange(num_points)
        
        removed = 0
        
        while removed < num_to_remove and len(available_indices) > 0:
            # Escolhe um ponto semente aleatório
            seed_idx = rng.choice(available_indices)
            
            # Encontra um número maior de vizinhos (cluster maior)
            K = rng.integers(5, 64)  # Cluster de tamanho 5 a 15 pontos
            distances, neighbor_indices = tree.query(pc[seed_idx], k=min(K, len(available_indices)))
            
            # Garante que neighbor_indices seja iterável
            if not isinstance(neighbor_indices, np.ndarray):
                neighbor_indices = [neighbor_indices]
            
            # Remove os pontos do cluster que ainda não foram removidos
            to_remove = []
            for n_idx in neighbor_indices:
                if mask[n_idx]:
                    to_remove.append(n_idx)
                    removed += 1
                    if removed >= num_to_remove:
                        break
            
            # Atualiza a máscara e os índices disponíveis
            mask[to_remove] = False
            available_indices = np.where(mask)[0]
        
        # Aplica a máscara para obter a nuvem corrompida
        corrupted_pc = pc[mask]
        return corrupted_pc
    
    def save_corrupted_data(self):
        """Salva todas as nuvens de pontos corrompidas como arquivos PLY."""
        for idx in range(len(self.data)):
            original_pc = self.data[idx]
            corrupted_pc = self.corrupt_pointcloud(original_pc, idx)
            filename_o = os.path.join(self.corrupted_dir, f"origin_{idx}.ply")
            filename = os.path.join(self.corrupted_dir, f"corrupted_{idx}.ply")
            self.write_ply(filename, corrupted_pc)
            self.write_ply(filename_o, original_pc)
    
    def write_ply(self, filename, pc):
        """Escreve a nuvem de pontos em formato PLY."""
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {pc.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "end_header"
        ]
        with open(filename, 'w') as f:
            f.write('\n'.join(header) + '\n')
            for point in pc:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    def __getitem__(self, idx):
        original_pc = self.data[idx]  # (2048, 3), já normalizado
        
        if self.mode == 'encoder':
            # Corrompe e amostra pontos (tamanho variável)
            corrupted_pc = self.corrupt_pointcloud(original_pc, idx)
            
            # Preenche com zeros para manter tamanho fixo
            padded_pc = np.zeros((self.num_points, 3))
            padded_pc[:corrupted_pc.shape[0]] = corrupted_pc
            return torch.tensor(padded_pc).float(), torch.tensor(original_pc).float()
        else:
            return torch.tensor(original_pc).float()