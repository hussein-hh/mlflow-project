import requests
import json
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names.tolist()

X_sample = X[:10]
y_sample = y[:10]

scaler = StandardScaler()
X_sample = scaler.fit_transform(X_sample)

mlflow.set_experiment("model_monitoring")

with mlflow.start_run(run_name="monitor_rf_live"):
    preds = []
    for i, x in enumerate(X_sample):
        payload = {
            "instances": [x.tolist()]
        }

        response = requests.post(
            "http://127.0.0.1:1234/invocations",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        pred = json.loads(response.text)["predictions"][0]
        preds.append(pred)

        print(f"Row {i+1}: pred={pred}, actual={y_sample[i]}")

    acc = accuracy_score(y_sample, preds)
    mlflow.log_metric("live_accuracy", acc)
    print(f"\n✅ Logged live accuracy: {acc:.4f}")
