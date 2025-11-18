# MLflow Integration Guide for Kedro Project

## 🚀 Quick Start

This project uses **MLflow** for experiment tracking, model versioning, and deployment alongside **Kedro** for pipeline orchestration.

### Prerequisites
- Python 3.8+
- Kedro project initialized
- Virtual environment activated

### Installation
```bash
# Install MLflow (if not already installed)
pip install mlflow

# Verify installation
mlflow --version
```

## 📊 MLflow Components in This Project

### 1. **Experiment Tracking**
All model training runs are automatically logged to MLflow with:
- Parameters (hyperparameters, configuration)
- Metrics (accuracy, loss, custom metrics)
- Artifacts (models, plots, data samples)
- Source code version

### 2. **Model Registry**
Production-ready models are registered and versioned for deployment.

### 3. **MLflow UI**
Web interface for visualizing and comparing experiments.

## 🎯 Usage

### Starting MLflow UI
```bash
# From project root directory
mlflow ui

# Custom port (default is 5000)
mlflow ui --port 8080

# Point to specific tracking URI
mlflow ui --backend-store-uri file:./mlruns
```

Access the UI at: http://localhost:5000

### Running Kedro Pipeline with MLflow Tracking

```bash
# Run the complete pipeline (MLflow tracking automatic)
kedro run

# Run specific pipeline
kedro run --pipeline=data_science

# Run with different parameters
kedro run --params n_estimators:200,max_depth:15
```

### Viewing Experiments in MLflow

1. Open MLflow UI: `mlflow ui`
2. Navigate to "Experiments" tab
3. Click on your experiment (default: "kedro-experiment")
4. Compare runs by selecting multiple runs and clicking "Compare"

## 📁 Project Structure with MLflow

```
your-kedro-project/
├── conf/
│   └── base/
│       ├── catalog.yml      # Data catalog configuration
│       └── parameters.yml   # Model parameters & MLflow config
├── data/
│   └── 06_models/          # Saved model artifacts
├── mlruns/                  # MLflow tracking data (auto-created)
│   ├── 0/                  # Default experiment
│   └── <experiment-id>/    # Your experiments
├── src/
│   └── pipelines/
│       └── data_science/
│           └── nodes.py     # MLflow tracking in train functions
└── notebooks/
    └── mlflow_analysis.ipynb  # Experiment analysis
```

## 🔧 Configuration

### Setting Tracking URI

**Local File System (Default)**
```python
# In your nodes.py or configuration
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
```

**Remote Tracking Server**
```python
mlflow.set_tracking_uri("http://your-mlflow-server:5000")
```

**Environment Variable**
```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
kedro run
```

### Configuring Experiments

```python
# In your training node
import mlflow

# Set experiment name
mlflow.set_experiment("my-experiment")

# Add tags for organization
mlflow.set_tags({
    "team": "data-science",
    "project": "customer-churn",
    "version": "v1.2"
})
```

## 📈 Common MLflow Commands

### Serving Models
```bash
# Serve the latest model version
mlflow models serve -m "models:/kedro_model/latest" -p 5001

# Serve specific run's model
mlflow models serve -m "runs:/<run-id>/model" -p 5001

# Test the served model
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"columns": ["feature1", "feature2"], "data": [[1.0, 2.0]]}'
```

### Model Registry Operations
```bash
# Register a model from a run
mlflow register-model \
  --run-id <run-id> \
  --model-path "model" \
  --name "production-model"

# Transition model stage
mlflow models transition-model-version-stage \
  --name "production-model" \
  --version 1 \
  --stage "Production"
```

### Comparing Experiments
```python
import mlflow
import pandas as pd

# Get experiment by name
experiment = mlflow.get_experiment_by_name("kedro-experiment")

# Load all runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Find best model
best_run = runs.loc[runs['metrics.test_accuracy'].idxmax()]
print(f"Best run ID: {best_run['run_id']}")
print(f"Best accuracy: {best_run['metrics.test_accuracy']}")
```

## 🎨 Best Practices

### 1. **Organize Experiments**
```python
# Use descriptive experiment names
mlflow.set_experiment(f"experiment_{datetime.now().strftime('%Y%m%d')}")
```

### 2. **Log Everything Important**
```python
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model_type": "RandomForest",
        "data_version": "v2.1",
        "feature_engineering": "standard_scaling"
    })
    
    # Log metrics at multiple steps
    for epoch in range(epochs):
        mlflow.log_metric("loss", loss, step=epoch)
    
    # Log artifacts
    mlflow.log_artifact("plots/confusion_matrix.png")
    mlflow.log_artifact("data/feature_importance.csv")
```

### 3. **Use Model Signatures**
```python
from mlflow.models.signature import infer_signature

# Infer signature from training data
signature = infer_signature(X_train, model.predict(X_train))

mlflow.sklearn.log_model(
    model, 
    "model",
    signature=signature
)
```

### 4. **Tag Runs for Easy Filtering**
```python
mlflow.set_tags({
    "git_commit": git_commit_hash,
    "dataset": "customer_data_2024",
    "preprocessed": "true",
    "model_type": "classifier"
})
```

## 🔍 Debugging & Troubleshooting

### View All Runs
```python
import mlflow
import pandas as pd

# Get all runs from all experiments
all_runs = mlflow.search_runs(search_all_experiments=True)
print(all_runs[['experiment_id', 'run_id', 'status', 'metrics.test_accuracy']])
```

### Clean Up Old Runs
```bash
# Delete specific run
mlflow runs delete --run-id <run-id>

# Delete entire experiment
mlflow experiments delete --experiment-id <experiment-id>
```

### Common Issues

**Issue: MLflow UI not showing runs**
```bash
# Check tracking URI
echo $MLFLOW_TRACKING_URI

# Ensure you're in the right directory
cd /path/to/kedro/project
mlflow ui
```

**Issue: Model serving fails**
```bash
# Check model flavor and dependencies
mlflow models show -m "models:/model_name/version"

# Ensure all dependencies are installed
pip install -r requirements.txt
```

## 🚢 Deployment Options

### 1. **Local REST API**
```bash
mlflow models serve -m "models:/kedro_model/Production" -p 5001
```

### 2. **Docker Container**
```bash
mlflow models build-docker \
  -m "models:/kedro_model/1" \
  -n "kedro-model-server"

docker run -p 5001:8080 kedro-model-server
```

### 3. **Cloud Deployment**
```bash
# AWS SageMaker
mlflow sagemaker deploy \
  -m "models:/kedro_model/1" \
  --region us-west-2 \
  --mode replace

# Azure ML
mlflow azureml deploy \
  -m "models:/kedro_model/1" \
  --workspace-name my-workspace
```

## 📚 Advanced Features

### Hyperparameter Tuning with MLflow
```python
import mlflow
from sklearn.model_selection import GridSearchCV

def hyperparameter_tuning(X_train, y_train):
    with mlflow.start_run(run_name="hyperparameter_search"):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(),
            param_grid,
            cv=5
        )
        
        grid_search.fit(X_train, y_train)
        
        # Log all combinations
        for i, params in enumerate(grid_search.cv_results_['params']):
            with mlflow.start_run(nested=True, run_name=f"combination_{i}"):
                mlflow.log_params(params)
                mlflow.log_metric("cv_score", grid_search.cv_results_['mean_test_score'][i])
        
        # Log best model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
        mlflow.log_params(grid_search.best_params_)
        
    return grid_search.best_estimator_
```

### Custom Metrics and Artifacts
```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def log_model_diagnostics(model, X_test, y_test):
    """Log additional model diagnostics to MLflow."""
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    
    # Log to MLflow
    mlflow.log_artifact('confusion_matrix.png')
    
    # Log custom metrics
    mlflow.log_metric("true_positives", cm[1, 1])
    mlflow.log_metric("false_positives", cm[0, 1])
    mlflow.log_metric("true_negatives", cm[0, 0])
    mlflow.log_metric("false_negatives", cm[1, 0])
```

## 📋 Workflow Example

```bash
# 1. Start MLflow UI in background
mlflow ui &

# 2. Run your Kedro pipeline
kedro run

# 3. View results in browser
open http://localhost:5000

# 4. Compare multiple runs
# Select runs in UI and click "Compare"

# 5. Deploy best model
mlflow models serve -m "runs:/<best-run-id>/model" -p 5001

# 6. Test deployment
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d @test_data.json
```

## 🔗 Useful Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kedro Documentation](https://kedro.readthedocs.io/)
- [MLflow Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)

## 💡 Tips

1. **Always use context managers** (`with mlflow.start_run()`) to ensure runs are properly closed
2. **Log early and often** - It's better to have too much information than too little
3. **Use meaningful run names** to easily identify experiments later
4. **Version your data** alongside your models for full reproducibility
5. **Set up a remote tracking server** for team collaboration

---

## Need Help?

- Check MLflow logs: `cat mlruns/mlflow-tracking.log`
- Verify setup: `mlflow doctor`
- Join MLflow Slack: [mlflow.slack.com](https://mlflow.slack.com)

Happy experimenting! 🎯
