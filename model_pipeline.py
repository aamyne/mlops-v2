# model_pipeline.py
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data():
    """Load and preprocess data."""
    df_80 = pd.read_csv("churn-bigml-80.csv")
    df_20 = pd.read_csv("churn-bigml-20.csv")

    categorical_features = ["State", "International plan", "Voice mail plan"]
    encoder = OneHotEncoder(cols=categorical_features, use_cat_names=True)
    df_80_encoded = encoder.fit_transform(df_80[categorical_features])
    df_20_encoded = encoder.transform(df_20[categorical_features])

    df_80 = df_80.drop(columns=categorical_features).join(df_80_encoded)
    df_20 = df_20.drop(columns=categorical_features).join(df_20_encoded)

    X_train, y_train = df_80.drop(columns=["Churn"]), df_80["Churn"]
    X_test, y_test = df_20.drop(columns=["Churn"]), df_20["Churn"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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


def save_model(model, encoder, scaler, filename="bagging_model.joblib"):
    """Save the model and preprocessing objects."""
    joblib.dump({"model": model, "encoder": encoder, "scaler": scaler}, filename)
    print("Model saved as", filename)


def load_model(filename="bagging_model.pkl"):
    """Load the saved model and preprocessing objects."""
    data = joblib.load(filename)
    return data["model"], data["encoder"], data["scaler"]
