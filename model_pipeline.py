# model_pipeline.py
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from category_encoders import OrdinalEncoder
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data():
    """Load and preprocess data."""
    df_80 = pd.read_csv("churn-bigml-80.csv")
    df_20 = pd.read_csv("churn-bigml-20.csv")

    # Drop redundant features (if they exist)
    redundant_features = [
        "Total day charge",
        "Total eve charge",
        "Total night charge",
        "Total intl charge",
    ]
    df_80 = df_80.drop(columns=redundant_features, errors="ignore")
    df_20 = df_20.drop(columns=redundant_features, errors="ignore")

    categorical_features = ["State", "International plan", "Voice mail plan"]
    encoder = OrdinalEncoder(cols=categorical_features)
    df_80_encoded = encoder.fit_transform(df_80)
    df_20_encoded = encoder.transform(df_20)

    X_train, y_train = df_80_encoded.drop(columns=["Churn"]), df_80_encoded["Churn"]
    X_test, y_test = df_20_encoded.drop(columns=["Churn"]), df_20_encoded["Churn"]

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Display the DataFrame
    print("\nX_train_scaled DataFrame:")
    print(X_train_scaled.head())  # Show first 5 rows for verification

    return X_train_scaled, y_train, X_test_scaled, y_test, encoder, scaler


def train_model(X_train, y_train, n_estimators=50, random_state=42):
    """Train a BaggingClassifier model."""
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return acc, report


def save_model(model, encoder, scaler, filename="churn_model.joblib"):
    """Save the model and preprocessing objects."""
    joblib.dump({"model": model, "encoder": encoder, "scaler": scaler}, filename)
    print("Model saved as", filename)


def load_model(filename="churn_model.joblib"):
    """Load the saved model and preprocessing objects."""
    data = joblib.load(filename)
    return data["model"], data["encoder"], data["scaler"]
