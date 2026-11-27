from pathlib import Path
import joblib

TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "MARVEL_Character_Classifier"
MODEL_ALIAS = "champion"
FALLBACK_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "logistic_regression_airflow.joblib"

def load_champion():
try:
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri(TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
return mlflow.sklearn.load_model(model_uri)
except Exception as error:
print(f"MLflow no disponible ({error}). Usando respaldo local...")
if not FALLBACK_MODEL_PATH.exists():
raise FileNotFoundError(f"No se encontro el modelo en: {FALLBACK_MODEL_PATH}")
return joblib.load(FALLBACK_MODEL_PATH)

def predict_dialogue(model, dialogue):
return model.predict([dialogue])[0]
