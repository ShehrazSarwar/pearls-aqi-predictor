# 🌍 Automated MLOps for Air Quality: Pearls AQI Predictor

[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/ShehrazSarwar/pearls-aqi-predictor)


## 📝 Introduction

This repository implements a production-grade, end-to-end MLOps ecosystem designed to forecast PM2.5 concentrations in Karachi, Pakistan. By integrating real-time environmental data with automated machine learning workflows, the system provides reliable **24, 48, and 72-hour air quality horizons**.

Built for scalability and resilience, the architecture leverages a **serverless stack** using GitHub Actions for orchestration, MongoDB Atlas as a high-performance Feature Store, and MLflow (via DagsHub) for comprehensive experiment tracking and a **"Champion vs. Challenger"** model registry. This setup ensures that the system doesn't just predict air quality, but continuously evolves and optimizes its accuracy without manual intervention.
## Key Features

*   **Automated Data Pipeline:** Hourly data ingestion from the Open-Meteo API for weather and air quality metrics.
*   **Advanced Feature Engineering:** Creates a rich feature set including time-based lags, rolling statistics, cyclical features, and interaction terms.
*   **Automated Model Training & Selection:** Daily training of three models (XGBoost, LightGBM, Random Forest) and automatic selection of the best-performing one based on R² and RMSE metrics.
*   **Model Registry & Promotion:** The winning model is registered in the MLflow Model Registry. A promotion script then compares it against the current "champion" model, promoting it if performance is superior or if the current model is stale.
*   **CI/CD Automation:**
    *   A GitHub Actions workflow runs every hour to update the dataset.
    *   A second workflow runs daily to retrain, evaluate, and potentially promote a new model.
*   **Prediction & Visualization:** Notebooks are provided to load the production model, generate forecasts, and validate performance against actuals.

## ⚙️ How It Works

The project is orchestrated by two main automated pipelines managed by GitHub Actions.

### 1. Hourly Data Pipeline (`hourly_data.yml`)

This workflow runs at the beginning of every hour to ensure the dataset is always fresh.

1.  **`data_extraction.py`**: Fetches the latest hourly air quality (`pm2_5`, `co`, `no2`, etc.) and weather (`temperature`, `wind_speed`, etc.) data from the Open-Meteo API. It performs an incremental update, adding only new records to a `raw_data` collection in MongoDB.
2.  **`feature_engineering.py`**: Reads the latest raw data, performs extensive feature engineering, and generates target variables (`target_h24`, `target_h48`, `target_h72`). The engineered data is then upserted into a `feature_store` collection in MongoDB, ready for model training.

### 2. Daily Model Pipeline (`daily_model.yml`)

This workflow runs once a day to train and manage the production model.

1.  **`model_train.py`**:
    *   Loads the complete dataset from the MongoDB `feature_store`.
    *   Trains three competing models: `XGBoost`, `LightGBM`, and `RandomForest`, each wrapped in a `MultiOutputRegressor` to handle the three prediction horizons.
    *   Evaluates the models on a validation set.
    *   Selects the best model (the "winner") based on the highest average R² score and lowest RMSE.
    *   Logs all parameters, metrics, and the model artifact to an MLflow experiment on DagsHub.
    *   Retrains the winning model on the *full* dataset and registers it as a new version in the `AQI_MultiOutput_Predictor` model registry.
2.  **`promote_model.py`**:
    *   Fetches the latest model version from the registry and the current model with the `champion` alias.
    *   Compares their R² scores.
    *   Promotes the new version to `champion` if its R² score is higher OR if the current champion model is more than 3 days old. This ensures both performance and model freshness.

## 📂 Repository Structure

```
.
pearls-aqi-predictor/
├── .github/
│   └── workflows/
│       ├── daily_model.yml             # Orchestrates model training and promotion every 24h
│       └── hourly_data.yml            # Orchestrates feature pipeline runs every hour
├── .gitignore
├── AQI_predict.pdf
├── automation_scripts/                 # Pipeline entry points for automation
│   ├── .cache.sqlite
│   ├── daily_model_pipeline.py         # Entry point for the Daily Training Pipeline
│   └── hourly_data_pipeline.py        # Entry point for the Hourly Feature Pipeline
├── models/
│   └── aqi_multi_output_model.pkl      # Serialized production model artifact
├── notebooks/                          # Research and Interpretability
│   ├── exploratory_data_analysis.ipynb # EDA to identify trends and patterns
│   ├── models/
│   │   └── aqi_multi_output_model.pkl
│   └── SHAP_feature_importance.ipynb   # Global and local model explainability
├── requirements.txt                    # Project dependencies
├── scripts/                            # Core Logic Modules
│   ├── .cache.sqlite
│   ├── data_extraction.py              # Logic to fetch raw API data
│   ├── feature_engineering.py          # Logic for transformations and targets
│   ├── model_train.py                 # Core model training and evaluation logic
│   └── promote_model.py               # Champion vs. Challenger promotion logic
├── test_notebooks/
│   ├── models/
│   │   └── aqi_multi_output_model.pkl
│   ├── Predict_AQI.ipynb
│   └── trained_model_validation.ipynb
└── test_scripts/                       # Experimental Development
    ├── db_ttl_setup.py
    ├── model_train.py
    ├── tensorflow_lstm_test.py         # Testing for deep learning architectures
    └── testdb.py
```

## Setup and Usage

### Prerequisites

*   Python 3.12 or later
*   A MongoDB database instance
*   A DagsHub account for MLflow tracking (or another MLflow server)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ShehrazSarwar/pearls-aqi-predictor.git
    cd pearls-aqi-predictor
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add the following credentials. This file is ignored by Git.

    ```env
    # MongoDB Credentials
    MONGO_URI="mongodb+srv://<user>:<password>@<cluster-uri>/"
    DB_NAME="aqi_predictor"

    # DagsHub / MLflow Credentials
    MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
    MLFLOW_TRACKING_PASSWORD="<your_dagshub_token>"
    MLFLOW_TRACKING_URI="<your_dagshub_url>"
    ```

### Manual Pipeline Execution

You can run the pipelines manually from your local machine.

1.  **Run the data pipeline** (fetches data and engineers features):
    ```bash
    python automation_scripts/hourly_data_pipeline.py
    ```

2.  **Run the model pipeline** (trains, evaluates, and registers the model):
    ```bash
    python automation_scripts/daily_model_pipeline.py
    ```
## 🏁 Conclusion & Future Roadmap

This project demonstrates a robust, production-ready MLOps architecture for environmental forecasting. By decoupling data ingestion from model retraining, the system ensures high data availability and continuous model improvement without manual intervention. The integration of **SHAP** for interpretability and **MLflow** for lifecycle management elevates it from a simple script to a professional machine learning service.

> **🚧 Project Status:** This project is currently in **active development**. While the backend automation and MLOps pipelines are fully functional, the **interactive web application** (Streamlit/Gradio dashboard) is currently being developed to provide a user-friendly interface for real-time predictions.

## 👤 Author & Credits

Developed with ❤️ by **Shehraz Sarwar Ghouri** as part of the **10Pearls Shine Program 2026 (Cohort 7)** - *Data Science*.


* **LinkedIn:** [Shehraz Sarwar](https://www.linkedin.com/in/shehraz-sarwar-ghouri-321394247/)
* **Program:** 10Pearls Shine Internship
* **Track:** Data Science & MLOps
* **Organization:** [10Pearls](https://10pearls.com/)

---

*If you find this project useful, please consider giving it a ⭐ on GitHub!*
