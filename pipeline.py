# src/your_project/pipeline.py
from kedro.pipeline import Pipeline, node
import mlflow

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        return model

pipeline = Pipeline([
    node(train_model,
         inputs=["X_train", "y_train"],
         outputs="model")
])
