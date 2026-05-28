import os
import torch
from PIL import Image
from torchvision import transforms
from model import get_model

def load_model(model_path, device):
    model = get_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_yield(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        prediction = model(image)

    yield_value = round(float(prediction.item()), 2)
    return yield_value


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)

    # test with first image in data folder
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    test_image = os.path.join(DATA_DIR, 'DJI_0006-h30.JPG')

    result = predict_yield(test_image, model, device)
    print(f"Predicted yield: {result} tons/hectare")