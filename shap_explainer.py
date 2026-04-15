import shap
import torch
import numpy as np
from model import DDoSModel

# Load model
model = DDoSModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

def explain(sample):
    sample = np.array(sample)

    explainer = shap.Explainer(lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy(), sample)
    shap_values = explainer(sample)

    return shap_values