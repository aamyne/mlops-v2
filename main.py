import mlflow
import time
import mlflow.system_metrics
import numpy as np
import psutil  # Import psutil for system metrics
from sklearn.ensemble import BaggingClassifier
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def get_system_metrics():
    """Returns system resource usage metrics."""
    return {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_used_percent": psutil.virtual_memory().percent,
        "disk_used_percent": psutil.disk_usage("/").percent,
    }


def main():
    # Configure MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Churn Prediction - BaggingClassifier")
    mlflow.enable_system_metrics_logging()

    with mlflow.start_run(run_name="BaggingClassifier Baseline"):
        time.sleep(15)

        print("\n=== Logging system metrics before training ===")
        mlflow.log_metrics(get_system_metrics())

        print("\n=== Preparing data ===")
        X_train, y_train, X_test, y_test, encoder, scaler = prepare_data()

        mlflow.log_params(
            {
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "num_features": X_train.shape[1],
                "class_ratio": f"{sum(y_train)/len(y_train):.2f}",
            }
        )

        print("\n=== Training BaggingClassifier model ===")
        model_params = {"n_estimators": 50, "random_state": 42}
        model = train_model(X_train, y_train, **model_params)

        mlflow.log_params(model_params)

        print("\n=== Logging system metrics after training ===")
        mlflow.log_metrics(get_system_metrics())

        print("\n=== Evaluating model ===")
        accuracy, classification_report = evaluate_model(model, X_test, y_test)

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": classification_report["weighted avg"]["precision"],
                "recall": classification_report["weighted avg"]["recall"],
                "f1_score": classification_report["weighted avg"]["f1-score"],
            }
        )

        print("\n=== Logging system metrics after evaluation ===")
        mlflow.log_metrics(get_system_metrics())

        print("\n=== Saving model ===")
        save_model(model, encoder, scaler, "churn_model.joblib")
        mlflow.log_artifact("churn_model.joblib")

        print("\n=== Loading saved model ===")
        loaded_model, loaded_encoder, loaded_scaler = load_model("churn_model.joblib")

        print("\n=== Sample prediction ===")
        sample_idx = 0
        sample_data = X_test[sample_idx].reshape(1, -1)
        sample_pred = loaded_model.predict(sample_data)
        print(f"Predicted: {sample_pred[0]} | Actual: {y_test.iloc[sample_idx]}")

        mlflow.log_dict(
            {
                "sample_prediction": {
                    "features": X_test[sample_idx].tolist(),
                    "prediction": int(sample_pred[0]),
                    "actual": int(y_test.iloc[sample_idx]),
                }
            },
            "sample_prediction.json",
        )

        print("\n=== Logging final system metrics ===")
        mlflow.log_metrics(get_system_metrics())

        print("\n=== MLflow Run ID ===")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("\n=== Workflow completed! ===")


if __name__ == "__main__":
    main()
