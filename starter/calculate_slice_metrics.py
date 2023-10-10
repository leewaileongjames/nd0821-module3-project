"""
Uses slice_metrics function from model.py to evaluate model on data slices
"""

import src.data as dt
import src.model as md
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


# Import and Clean data
df = pd.read_csv('data/census.csv')
df = dt.clean_data(df)

# Import model training outputs
model = joblib.load('training_artifacts/model.pkl')
encoder = joblib.load('training_artifacts/encoder.pkl')
lb = joblib.load('training_artifacts/label_binarizer.pkl')

# Get test data
_, test = train_test_split(df, test_size=0.20, random_state=42)

# Transform data
cat_features = test.select_dtypes(include=object).columns.tolist()
cat_features.remove('salary')

# Clears any existing content in the file
with open('slice_output.txt', 'w') as file:
    pass


for feature in cat_features:
    with open('slice_output.txt', 'a') as file:
        file.write(f"\n\n=== Feature: {feature} ===\n")

    for category in test[feature].unique():
        test_slice = test[ test[feature] == category ]
        X_test_slice, y_test_slice, _, _ = dt.process_data(
                                                test_slice,
                                                categorical_features=cat_features,
                                                label='salary',
                                                encoder=encoder,
                                                lb=lb,
                                                training=False)

        test_slice_preds = md.inference(model, X_test_slice)
        precision_slice, recall_slice, fbeta_slice = md.compute_model_metrics(
            y_test_slice,
            test_slice_preds)
        
        with open(f'slice_output.txt', 'a') as file:
            file.write(f"-- {category} --\n")
            file.write(f"Precision: {precision_slice}\n")
            file.write(f"Recall: {recall_slice}\n")
            file.write(f"FBeta: {fbeta_slice}\n\n")
    print(f"Calaulation for {feature} complete ...")



