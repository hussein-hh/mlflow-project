import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import warnings
warnings.filterwarnings("ignore")

mlflow.set_experiment("rf_hyperopt_breast_cancer")

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

def objective(params):
    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            random_state=42,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

        return {"loss": -acc, "status": STATUS_OK}

search_space = {
    "n_estimators": hp.quniform("n_estimators", 50, 200, 10),
    "max_depth": hp.quniform("max_depth", 3, 15, 1),
}

with mlflow.start_run(run_name="rf_hyperopt_parent"):
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials,
    )
    mlflow.log_params(best_result)
