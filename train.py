import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.custom_dataset import ModelNet40Sphere
from model.vae import PointVAE, chamfer_loss
import numpy as np
from tqdm import tqdm
import h5py
import os

# Configurações globais
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# Hiperparâmetros
BATCH_SIZE = 64//4
LATENT_DIM = 256
EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
CORRUPTION_RATE = 0.6

# Paths
ROOT_DIR = "data/modelnet40_ply_hdf5_2048"
TRAIN_FILE_LIST = "train_files.txt"

def main():
    # Dataset e DataLoader
    train_dataset = ModelNet40Sphere(
        root_dir=ROOT_DIR,
        file_list=TRAIN_FILE_LIST,
        mode='encoder',
        # corruption_rate=CORRUPTION_RATE
    )
    # train_dataset.save_corrupted_data()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Modelo
    model = PointVAE(latent_dim=LATENT_DIM).to(DEVICE)
    
    # Otimização
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        verbose=True
    )

    # Loop de treino
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for corrupted, original in progress_bar:
            # Mover dados para GPU
            corrupted = corrupted.transpose(1, 2).to(DEVICE)  # (B, 3, N)
            original = original.to(DEVICE)  # (B, 2048, 3)

            # Forward pass
            optimizer.zero_grad()
            recon, mu, logvar = model(corrupted)

            # Calcular perdas
            cd_loss = chamfer_loss(recon, original)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = cd_loss + 0.1 * kl_loss

            # Backpropagation
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Atualizar métricas
            epoch_loss += total_loss.item()
            progress_bar.set_postfix({
                "Loss": total_loss.item(),
                "CD Loss": cd_loss.item(),
                "KL Loss": kl_loss.item()
            })

        # Ajustar learning rate e salvar modelo
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Salvar checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"best_vae_model.pth")
            
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"checkpoint_epoch_{epoch+1}.pth")

        print(f"\nEpoch {epoch+1} Completa | Loss Médio: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}\n")

    # Salvar modelo final
    torch.save(model.state_dict(), "final_vae_model.pth")
    print("Treino concluído!")

if __name__ == "__main__":
    main()