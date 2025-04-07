from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from torchvision import transforms
import your_model_loader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = your_model_loader.load_model()
class_names = your_model_loader.class_names

nutrition_data = {
    "nasi goreng": {"kalori": 250, "protein": 6, "lemak": 8, "karbo": 35},
    "rendang": {"kalori": 280, "protein": 20, "lemak": 15, "karbo": 10},
    "sate ayam": {"kalori": 200, "protein": 18, "lemak": 10, "karbo": 8},
    "gado-gado": {"kalori": 180, "protein": 8, "lemak": 9, "karbo": 22},
    "soto ayam": {"kalori": 150, "protein": 12, "lemak": 6, "karbo": 10},
    "bakso": {"kalori": 220, "protein": 15, "lemak": 12, "karbo": 15},
    "mie goreng": {"kalori": 320, "protein": 9, "lemak": 14, "karbo": 40}
}

def predict_food(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    return class_names[pred.item()]

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    label = predict_food(img)
    nutrition = nutrition_data.get(label, {})

    return {"label": label, "nutrition": nutrition}
