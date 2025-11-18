import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, X_test, y_test, parameters):
    """Train model with MLflow tracking."""

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(parameters)

        # Train model
        model = RandomForestClassifier(
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            random_state=parameters["random_state"]
        )
        model.fit(X_train, y_train)

        # Make predictions
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Log metrics
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="kedro_model"
        )

        logger.info(f"Model trained. Test accuracy: {test_score:.3f}")

    return model
