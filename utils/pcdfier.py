import os
import trimesh
import numpy as np

def convert_mesh_to_pointcloud(mesh_path, num_points=2048):
    # Carrega a malha
    mesh = trimesh.load(mesh_path)
    
    # Amostra pontos uniformemente da superfície da malha
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    # Normaliza os pontos para a esfera unitária
    points -= np.mean(points, axis=0)  # Centraliza na origem
    points /= np.max(np.linalg.norm(points, axis=1))  # Escala para o raio 1
    
    return points

# Diretório de entrada (malhas) e saída (pontos)
input_dir = "data/ModelNet40"
output_dir = "data/ModelNet40_PointClouds"

# Processa todos os arquivos .off
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    # Cria diretório de saída
    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)
    
    for file in os.listdir(class_path):
        if file.endswith(".off"):
            mesh_path = os.path.join(class_path, file)
            points = convert_mesh_to_pointcloud(mesh_path)
            
            # Salva como .npy ou .txt
            output_path = os.path.join(output_class_dir, file.replace(".off", ".npy"))
            np.save(output_path, points)