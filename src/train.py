import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import CropYieldDataset, get_transforms
from model import CropYieldModel, get_model
import matplotlib.pyplot as plt

def train():
    # paths
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CSV_PATH = os.path.join(DATA_DIR, 'labels.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # dataset
    train_transform, val_transform = get_transforms()
    full_dataset = CropYieldDataset(CSV_PATH, DATA_DIR, transform=train_transform)

    # split into train and validation
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # apply val transform to val dataset
    val_dataset.dataset.transform = val_transform

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = get_model(device)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # training
        model.train()
        train_loss = 0.0
        for images, yields in train_loader:
            images = images.to(device)
            yields = yields.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, yields)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, yields in val_loader:
                images = images.to(device)
                yields = yields.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, yields)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
            print(f"  --> Best model saved!")

    # save final model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'final_model.pth'))
    print("\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")

    # plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'training_curve.png'))
    print("Training curve saved!")

if __name__ == "__main__":
    train()