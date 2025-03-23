import h5py
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dota_dataset import Dota2Dataset
from models import DenseNet
from train_utils import split_indices

def train_densenet(train_loader, val_loader, input_dim, hidden_dim=128, lr=1e-3, epochs=2, device='cpu'):
    model = DenseNet(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for hero_slice, pos_enc, label in train_loader:
            hero_slice = hero_slice.to(device)
            pos_enc = pos_enc.to(device)
            label = label.float().to(device)

            # Combine the slice + pos_enc
            x = torch.cat([hero_slice, pos_enc], dim=1)  # shape (batch, 125+pos_dim)
            
            logits = model(x).squeeze(1)  # shape (batch,)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct = (preds == label.long()).sum().item()
            total_correct += correct
            total_samples += label.size(0)
            total_loss += loss.item() * label.size(0)

        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model

def evaluate(model, data_loader, device='cpu'):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for hero_slice, pos_enc, label in data_loader:
            hero_slice = hero_slice.to(device)
            pos_enc = pos_enc.to(device)
            label = label.float().to(device)

            x = torch.cat([hero_slice, pos_enc], dim=1)
            logits = model(x).squeeze(1)
            loss = criterion(logits, label)

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct = (preds == label.long()).sum().item()
            total_correct += correct
            total_samples += label.size(0)
            total_loss += loss.item() * label.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def main(fp='draft_data.h5'):
    # 1) Determine total N from the file
    with h5py.File(fp, 'r') as f:
        N = f['tensors'].shape[0]

    # 2) Split indices
    train_idx, val_idx, test_idx = split_indices(N, 0.8, 0.1, 0.1, seed=42)

    # 3) Create subsets
    pos_dim = 16
    train_dataset = Dota2Dataset(fp, indices=train_idx, positional_encoding_dim=pos_dim)
    val_dataset   = Dota2Dataset(fp, indices=val_idx,   positional_encoding_dim=pos_dim)
    test_dataset  = Dota2Dataset(fp, indices=test_idx,  positional_encoding_dim=pos_dim)

    # 4) Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # 5) Train the model
    # Our input dimension = 125 + 16
    input_dim = 125 + pos_dim
    model = train_densenet(train_loader, val_loader, input_dim, hidden_dim=128, lr=1e-3, epochs=20, device='cuda')

    # 6) Evaluate final test performance
    test_loss, test_acc = evaluate(model, test_loader, device='cuda')
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return model


if __name__ == '__main__':
    model = main("draft_data.h5").state_dict()
    torch.save(model, "first-densenet.pt")