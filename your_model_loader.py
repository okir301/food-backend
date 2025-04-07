import torch
from torchvision import models

def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 7)  # ganti sesuai jumlah kelas
    model.load_state_dict(torch.load("food_model.pth", map_location="cpu"))
    model.eval()
    return model

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
