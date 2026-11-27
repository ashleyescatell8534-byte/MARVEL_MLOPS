from pathlib import Path
import joblib

TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "MARVEL_Character_Classifier"
MODEL_ALIAS = "champion"

BASE_DIR = Path(__file__).resolve().parent.parent
FALLBACK_MODEL_PATH = BASE_DIR / "models" / "logistic_regression_airflow.joblib"
FALLBACK_VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer_airflow.joblib"


class LocalTextPredictor:
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model

    def predict(self, texts):
        features = self.vectorizer.transform(texts)
        return self.model.predict(features)


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
        if not FALLBACK_VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"No se encontro el vectorizador en: {FALLBACK_VECTORIZER_PATH}")
        model = joblib.load(FALLBACK_MODEL_PATH)
        vectorizer = joblib.load(FALLBACK_VECTORIZER_PATH)
        return LocalTextPredictor(vectorizer, model)


def predict_dialogue(model, dialogue):
    return model.predict([dialogue])[0]