import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data():
    df = pd.read_csv("data/masour-realestate.csv").dropna(subset=["final_rent_price_usd"])
    X = df.drop(columns=["final_rent_price_usd"])
    y = df["final_rent_price_usd"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_tr, num_cols),
        ("cat", cat_tr, cat_cols)
    ])

def objective(params):
    X_train, X_test, y_train, y_test = data_splits
    pre = create_preprocessor(X_train)
    model = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        random_state=42
    )
    pipeline = Pipeline([("prep", pre), ("model", model)])
    with mlflow.start_run(nested=True):
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(pipeline, "model")
        return {"loss": rmse, "status": STATUS_OK}

if __name__ == "__main__":
    mlflow.set_experiment("rf_hyperopt_realestate")
    data_splits = load_data()

    search_space = {
        "n_estimators": hp.quniform("n_estimators", 50, 300, 50),
        "max_depth":    hp.quniform("max_depth", 3, 20, 1)
    }

    with mlflow.start_run(run_name="rf_hyperopt_parent_realestate"):
        trials = Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials
        )

        best_int = {k: int(v) for k, v in best.items()}
        mlflow.log_params(best_int)
        print("Best hyperparameters:", best_int)
