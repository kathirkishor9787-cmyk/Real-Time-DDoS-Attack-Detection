import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create sample dataset
data = {
    "packet_count": [10,20,30,100,200,300],
    "byte_count": [1000,2000,3000,10000,20000,30000],
    "label": [0,0,0,1,1,1]
}

df = pd.DataFrame(data)

X = df[["packet_count", "byte_count"]]
y = df["label"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Training complete")
print("model.pkl saved successfully")