import requests, json
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def main():
    mlflow.set_experiment("realestate_monitoring")

    df = pd.read_csv("data/masour-realestate.csv").dropna(subset=["final_rent_price_usd"])
    X = df.drop(columns=["final_rent_price_usd"])
    y = df["final_rent_price_usd"]
    X_sample = X.iloc[:10]
    y_sample = y.iloc[:10]

    with mlflow.start_run(run_name="monitor_realestate_live"):
        preds = []
        for i, (_, row) in enumerate(X_sample.iterrows()):
            # send full-record format
            payload = {"dataframe_records": [row.to_dict()]}
            resp = requests.post(
                "http://127.0.0.1:1235/invocations",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            pred = resp.json()["predictions"][0]
            preds.append(pred)
            print(f"Row {i+1}: pred={pred:.2f}, actual={y_sample.iloc[i]:.2f}")

        rmse = np.sqrt(mean_squared_error(y_sample, preds))
        mlflow.log_metric("live_rmse", rmse)
        print(f"\n Logged live RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()
