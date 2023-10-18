"""
Train KNN model on census data, Save model, encoder, binarizer
"""

from sklearn.model_selection import train_test_split
from os.path import dirname
from os.path import abspath
import pandas as pd
import src.model as md
import src.data as dt
import joblib


if __name__ == '__main__':

    current_dir = dirname(abspath(__file__))

    # Import & Clean data
    df = pd.read_csv(f'{current_dir}/data/census.csv')
    df = dt.clean_data(df)

    # Split data
    train, test = train_test_split(df, test_size=0.20, random_state=42)

    # Get a list of categorical features
    cat_features = train.select_dtypes(include=object).columns.tolist()
    cat_features.remove('salary')

    X_train, y_train, encoder, lb = dt.process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    X_test, y_test, encoder, lb = dt.process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        training=False
    )

    # Train KNN model
    model = md.train_model(X_train, y_train)

    # Measure model's performance
    pred_y = md.inference(model, X_test)

    precision, recall, fbeta = md.compute_model_metrics(y_test, pred_y)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"FBeta: {fbeta}")

    # Save the model, encoder and label binarizer
    joblib.dump(model, f'{current_dir}/training_artifacts/model.pkl')
    joblib.dump(encoder, f'{current_dir}/training_artifacts/encoder.pkl')
    joblib.dump(lb, f'{current_dir}/training_artifacts/label_binarizer.pkl')
