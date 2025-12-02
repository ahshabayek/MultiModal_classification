# MultiModalClassification

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

A multimodal classification project using ViLBERT (Vision-and-Language BERT) for the Hateful Memes dataset. This project implements binary classification to detect hateful content in memes by combining image and text understanding.

## Data Loading

The project automatically loads data from HuggingFace (`neuralcatcher/hateful_memes`) and downloads images from Google Drive. This mirrors the approach in `notebooks/hateful-memes.ipynb`.

**Data flow:**
1. Load dataset from HuggingFace (train/validation/test splits)
2. Remove duplicates
3. Download image archive from Google Drive (~1GB)
4. Fetch any missing images from HuggingFace backup
5. Preprocess text and validate image paths

## Available Pipelines

The project provides multiple pipeline modes for different use cases:

| Pipeline | Command | Description |
|----------|---------|-------------|
| Data Processing | `kedro run --pipeline=data_processing` | Load data from HuggingFace and preprocess |
| Training | `kedro run --pipeline=vilbert_train` | Full pipeline: data loading + training + evaluation |
| Validation | `kedro run --pipeline=vilbert_validate` | Validate a locally trained model on the test set |
| Inference (Pretrained) | `kedro run --pipeline=vilbert_inference` | Run inference using pretrained HuggingFace weights |
| Inference (Local) | `kedro run --pipeline=vilbert_inference_local` | Run inference using locally trained weights |

**Model-only pipelines** (use if data is already processed):
- `kedro run --pipeline=model_training`
- `kedro run --pipeline=model_validation`
- `kedro run --pipeline=model_inference`
- `kedro run --pipeline=model_inference_local`

## MLflow Experiment Tracking

This project uses MLflow for experiment tracking via the `kedro-mlflow` plugin.

### What Gets Logged

**During Training:**
- Model parameters (total params, trainable params, device)
- Training configuration (learning rate, batch size, epochs, etc.)
- Per-epoch metrics: train_loss, val_loss, val_auroc, val_accuracy, val_f1
- Best model checkpoint

**During Inference/Evaluation:**
- Inference metrics: AUROC, accuracy, precision, recall, F1
- Confusion matrix values (TP, TN, FP, FN)
- Prediction validation checks

### Viewing MLflow UI

To view experiment results in the MLflow dashboard:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Start MLflow UI
mlflow ui --backend-store-uri mlruns
```

Then open http://localhost:5000 in your browser.

## Prediction Validation Checks

The inference pipelines include automatic validation checks on predictions:

| Check | Description |
|-------|-------------|
| `no_null_predictions` | Ensures no null values in predictions |
| `valid_binary_predictions` | Verifies predictions are 0 or 1 |
| `valid_probability_range` | Confirms probabilities are in [0, 1] |
| `no_null_probabilities` | Ensures no null probability values |
| `prediction_count` | Verifies predictions were generated |
| `class_distribution` | Checks for reasonable class balance |
| `better_than_random` | Validates AUROC > 0.5 (better than random) |

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the files `tests/test_run.py` and `tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
