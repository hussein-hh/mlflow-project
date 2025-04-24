import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def main():
    mlflow.set_experiment("realestate_baseline")

    df = pd.read_csv("data/masour-realestate.csv")
    df = df.dropna(subset=["final_rent_price_usd"])

    X = df.drop(columns=["final_rent_price_usd"])
    y = df["final_rent_price_usd"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ])

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42
        )),
    ])

    with mlflow.start_run(run_name="rf_baseline_realestate"):
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)

        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Logged real-estate baseline run with RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
