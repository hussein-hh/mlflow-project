import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5005")

run_id    = "070e7c5f147a436a88f6a3479f778cb1"
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name="realestate_rf_model",
    await_registration_for=60
)

print(f"Registered model '{result.name}' version {result.version}")
