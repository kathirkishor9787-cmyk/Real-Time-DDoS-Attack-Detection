import torch
from model import DDoSModel

model = DDoSModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict(data):
    data = torch.tensor(data, dtype=torch.float32)
    output = model(data)
    _, predicted = torch.max(output, 1)
    return predicted.item()