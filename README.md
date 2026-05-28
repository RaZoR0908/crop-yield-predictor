# Crop Yield Predictor

Predicts crop yield (tons/hectare) from drone field images using EfficientNet transfer learning.

**Demo:** Upload an image → `{"predicted_yield": 4.5, "unit": "tons/hectare"}`

## Project Structure
```
crop-yield-predictor/
├── data/               # Images + labels.csv
├── models/             # Trained weights
├── src/
│   ├── dataset.py      # Data loading
│   ├── model.py        # EfficientNet model
│   ├── train.py        # Training
│   ├── evaluate.py     # MAE/RMSE metrics
│   ├── predict.py      # Single image inference
│   └── generate_labels.py
├── api.py              # FastAPI server
└── requirements.txt
```

## Model
- **Architecture:** EfficientNet-B0 with regression head
- **Input:** 224×224 RGB drone images
- **Output:** Yield in tons/hectare
- **Metrics:** MAE = 0.69, RMSE = 0.85

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/RaZoR0908/crop-yield-predictor.git
cd crop-yield-predictor
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Download images from [Kaggle](https://www.kaggle.com/datasets/suhelahamed/drone-camera-image-dataset-of-agriculture-fields) and place all `.jpg` files in `data/` folder.

### 5. Generate labels
```bash
python src/generate_labels.py
```

### 6. Train the model
```bash
cd src
python train.py
```

### 7. Evaluate the model
```bash
python evaluate.py
```

### 8. Run the API
```bash
cd ..
python api.py
```

### 9. Test the API
Open browser → **http://localhost:8000/docs**
Upload a drone field image and get yield prediction!

## Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict crop yield |

## Stack
Python 3.11 · PyTorch · EfficientNet · FastAPI · Pillow · Scikit-learn