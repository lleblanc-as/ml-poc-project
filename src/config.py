from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT_DIR

ENV_FILE = ROOT_DIR / ".env"
SRC_DIR = ROOT_DIR / "src"
LOGS_DIR = ROOT_DIR / "logs"
SCRIPTS_DIR = ROOT_DIR / "scripts"
APP_ENTRYPOINT = SRC_DIR / "app.py"
STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"

MODELS = {
    "log_reg": {
        "name": "Logistic Regression",
        "description": "Modèle de classification simple pour prédire le risque cyclonique élevé aux Antilles.",
        "path": MODELS_DIR / "log_reg.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Modèle d'ensemble basé sur des arbres de décision pour détecter les observations cycloniques à haut risque.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
}

STREAMLIT_APP_TITLE = "Prédiction du risque cyclonique aux Antilles"
STREAMLIT_APP_DESCRIPTION = "Proof of Concept Machine Learning pour identifier les situations cycloniques à haut risque dans la zone Antilles."
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"