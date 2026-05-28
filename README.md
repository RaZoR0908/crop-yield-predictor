# Crop Yield Prediction using CNN

A deep learning system that predicts crop yield (tons/hectare) from drone aerial field images using EfficientNet transfer learning.

## Demo
- Input: Drone crop field image
- Output: `{"predicted_yield": 4.5, "unit": "tons/hectare"}`

## Project Structure
crop-yield-predictor/
├── data/                   # Dataset images + labels.csv
├── models/                 # Trained model weights
├── src/
│   ├── dataset.py          # Data loading and transforms
│   ├── model.py            # EfficientNet CNN model
│   ├── train.py            # Model training script
│   ├── evaluate.py         # MAE/RMSE evaluation
│   ├── predict.py          # Single image prediction
│   └── generate_labels.py  # Generate yield labels from images
├── api.py                  # FastAPI REST API
├── requirements.txt        # Python dependencies
└── README.md

## Model
- Architecture: EfficientNet-B0 with custom regression head
- Task: Regression (predicting yield in tons/hectare)
- Input: 224x224 RGB drone field images
- Output: Single yield value (tons/hectare)
- Evaluation: MAE = 0.69, RMSE = 0.85

## Dataset
Drone Camera Image Dataset of Agriculture Fields from Kaggle.
👉 https://www.kaggle.com/datasets/suhelahamed/drone-camera-image-dataset-of-agriculture-fields

Download and place all images in the `data/` folder.

## Setup Instructions

### 1. Clone the repository
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
Download images from Kaggle link above and place all JPG files in `data/` folder.

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
Open browser and go to:http://localhost:8000/docs
Upload a drone field image and get yield prediction!

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API status |
| GET | /health | Health check |
| POST | /predict | Predict crop yield from image |

## Sample API Response
```json
{
  "predicted_yield": 5.11,
  "unit": "tons/hectare",
  "status": "success"
}
```

## Tech Stack
- Python 3.11
- PyTorch + EfficientNet (Transfer Learning)
- FastAPI + Uvicorn
- Pillow, NumPy, Pandas
- Scikit-learn (evaluation metrics)