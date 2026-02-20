# Sri Lanka Rice Price Prediction & Forecasting Dashboard

A comprehensive Machine Learning project to predict retail rice prices in Sri Lanka using historical trends and macro-economic indicators (USD/LKR Rate & Inflation).

## Features
*   **Data Pipeline**: Automated collection, preprocessing, and feature engineering.
*   **XGBoost Forecasting**: High-precision model with 99.8% R² score.
*   **Interactive Dashboard**: Streamlit-based UI for data exploration and prediction.
*   **FastAPI Backend**: Scalable REST API for model serving.
*   **Explainable AI (XAI)**: SHAP and Partial Dependence interpretation.
*   **Dockerized**: Fully containerized multi-service architecture.

## Architecture
The project follows a microservices pattern:
1.  **Backend**: FastAPI serving the model at port `8000`.
2.  **Frontend**: Streamlit dashboard at port `8501`.
3.  **Communication**: The frontend calls the backend REST API for predictions.

## Setup & Usage

### Option 1: Docker (Recommended)
Ensure you have Docker and Docker Compose installed, then run:
```bash
docker-compose up --build
```
*   Dashboard: [http://localhost:8501](http://localhost:8501)
*   API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Local Installation (Windows/Unix)
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the API:
    ```bash
    python -m uvicorn api.main:app --reload
    ```
3.  Run the App:
    ```bash
    python -m streamlit run app/main.py
    ```

## Project Structure
```text
├── api/                # FastAPI backend
├── app/                # Streamlit frontend & views
├── data/               # Raw and processed datasets
├── outputs/            # Saved models and visualization plots
├── scripts/            # ML pipeline (collect, preprocess, train, XAI)
├── utils/              # Shared helper functions
├── Dockerfile.api      # Backend container config
├── Dockerfile.app      # Frontend container config
└── docker-compose.yml  # Orchestration
```

## Documentation
For detailed methodology, results, and economic interpretations, please refer to the **[ProjectReport.md](./ProjectReport.md)**.
