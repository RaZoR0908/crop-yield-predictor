import os
import sys
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

# add src to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import get_model
from predict import load_model, predict_yield

app = FastAPI(
    title="Crop Yield Prediction API",
    description="Upload a drone crop field image to predict yield in tons/hectare",
    version="1.0.0"
)

# load model on startup
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None

@app.on_event("startup")
async def startup_event():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, device)
        print("Model loaded successfully!")
    else:
        print("Warning: No trained model found. Train the model first!")

@app.get("/")
def root():
    return {"message": "Crop Yield Prediction API is running!"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # check if model is loaded
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Please train the model first!"}
        )

    # check file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return JSONResponse(
            status_code=400,
            content={"error": "Only JPG and PNG images are supported!"}
        )

    try:
        # read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # save temp image
        temp_path = os.path.join(BASE_DIR, 'temp_image.jpg')
        image.save(temp_path)

        # predict
        yield_value = predict_yield(temp_path, model, device)

        # cleanup temp file
        os.remove(temp_path)

        return {
            "predicted_yield": yield_value,
            "unit": "tons/hectare",
            "status": "success"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)