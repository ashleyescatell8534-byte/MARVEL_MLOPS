@"
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

# ==========================================================
# CONFIGURACION
# ==========================================================

TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "MARVEL_Character_Classifier"
MODEL_ALIAS = "champion"
FALLBACK_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "logistic_regression_airflow.joblib"


# ==========================================================
# CARGAR MODELO CHAMPION
# ==========================================================

def load_champion():
    print("=" * 60)
    print("MARVEL MLOPS - PREDICCION DE PERSONAJES")
    print("=" * 60)

    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        print("\nIntentando cargar modelo Champion desde MLflow...")
        model = mlflow.sklearn.load_model(model_uri)
        print("Modelo cargado desde MLflow correctamente.")
        return model
    except Exception as error:
        print(f"\nMLflow no disponible ({error}). Usando respaldo local...")
        if not FALLBACK_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo de respaldo en: {FALLBACK_MODEL_PATH}"
            )
        model = joblib.load(FALLBACK_MODEL_PATH)
        print(f"Modelo de respaldo cargado desde: {FALLBACK_MODEL_PATH}")
        return model


# ==========================================================
# REALIZAR PREDICCIONES
# ==========================================================

def predict_dialogue(model, dialogue):
    prediction = model.predict([dialogue])[0]
    return prediction


# ==========================================================
# INTERFAZ DE USUARIO
# ==========================================================

def main():
    model = load_champion()

    print("\nEscribe un dialogo de Marvel para identificar")
    print("al personaje que probablemente lo dijo.")
    print("Escribe 'salir' para terminar.")

    while True:
        print("\n" + "-" * 60)
        dialogue = input("Ingresa el dialogo: ").strip()

        if dialogue.lower() == "salir":
            print("\nPrograma finalizado.")
            break

        if not dialogue:
            print("Por favor, ingresa un dialogo valido.")
            continue

        try:
            character = predict_dialogue(model, dialogue)
            print("\nResultado de la prediccion:")
            print("Personaje:", character)
            print("\nNota: La prediccion puede no ser correcta.")
        except Exception as error:
            print("Ocurrio un error al realizar la prediccion:")
            print(error)


# ==========================================================
# EJECUCION
# ==========================================================

if __name__ == "__main__":
    main()
"@ | Out-File -FilePath src\predict.py -Encoding utf8
