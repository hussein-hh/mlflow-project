import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def plot_confusion(cm, path):
    plt.figure()
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    ticks = range(len(cm))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    for i in ticks:
        for j in ticks:
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    mlflow.set_experiment("breast_cancer_baseline")

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }

    with mlflow.start_run(run_name="rf_baseline"):
        mlflow.log_params(params)

        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(clf, "model")

        cm = confusion_matrix(y_test, preds)
        cm_path = "confusion_matrix.png"
        plot_confusion(cm, cm_path)
        mlflow.log_artifact(cm_path)

        print(f"Logged run with accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
