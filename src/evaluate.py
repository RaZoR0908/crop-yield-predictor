import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import CropYieldDataset, get_transforms
from model import get_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate():
    # paths
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CSV_PATH = os.path.join(DATA_DIR, 'labels.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pth')

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # dataset
    _, val_transform = get_transforms()
    full_dataset = CropYieldDataset(CSV_PATH, DATA_DIR, transform=val_transform)

    # same split as training
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size],
                                   generator=torch.Generator().manual_seed(42))

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # load model
    model = get_model(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully!")

    # evaluate
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, yields in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(yields.numpy())

    # metrics
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    print(f"\n--- Evaluation Results ---")
    print(f"MAE  (Mean Absolute Error): {mae:.4f} tons/hectare")
    print(f"RMSE (Root Mean Sq Error) : {rmse:.4f} tons/hectare")
    print(f"\nSample Predictions:")
    print(f"{'Real':<10} {'Predicted':<10}")
    print("-" * 20)
    for real, pred in zip(all_targets[:10], all_preds[:10]):
        print(f"{real:<10.2f} {pred:<10.2f}")

if __name__ == "__main__":
    evaluate()