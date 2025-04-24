import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5005")

run_id = "8596f26a1f98420da0bb2cca04f5da28"
model_name = "breast_cancer_rf_model"
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
    await_registration_for=60
)

print(f"Model registered as: {result.name}, version: {result.version}")
