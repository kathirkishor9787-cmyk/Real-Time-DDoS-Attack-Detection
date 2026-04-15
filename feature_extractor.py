import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.select_dtypes(include=['int64', 'float64'])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    return scaled_data