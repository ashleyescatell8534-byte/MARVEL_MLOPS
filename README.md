
# MARVEL_MLOPS

## Descripción del proyecto

Proyecto de Machine Learning enfocado en la clasificación de diálogos del Universo Cinematográfico de Marvel (MCU), con el objetivo de predecir qué personaje pronunció cada diálogo.

## Modelos utilizados

Este proyecto utiliza Regresión Logística, Random Forest y SVM.

También incluye un modelo BERT pequeño para comparar diferentes enfoques de clasificación de texto.

## Herramientas

- Python
- Pandas y NumPy
- Scikit-learn
- PyTorch y Transformers
- MLflow
- Git

## Dataset

Se utiliza un conjunto de diálogos del Universo Cinematográfico de Marvel.

## Evaluación de modelos

Los modelos se evaluaron mediante métricas de Accuracy, Precision, Recall y F1-score. Estas métricas permiten comparar el rendimiento de los algoritmos utilizados en la clasificación de diálogos.

## Modelo SVM

Se implementó un modelo Support Vector Machine (SVM) utilizando TF-IDF para transformar los diálogos en características numéricas y clasificar los personajes del Universo Cinematográfico de Marvel.

## Despliegue en producción (Render)

La aplicación de predicción de personajes de Marvel (FastAPI + interfaz web) se encuentra desplegada y operativa en:

- **URL pública:** `https://marvel-mlops.onrender.com`
- **Documentación interactiva (Swagger UI):** `https://marvel-mlops.onrender.com/docs`
- **Endpoint de predicción:** `POST https://marvel-mlops.onrender.com/predict`

### Arquitectura y respaldo en producción
- **Imagen de API aislada:** Construida mediante `Dockerfile.api` y `requirements-api.txt` para ejecutar FastAPI/Uvicorn sin dependencias pesadas de Airflow o MLflow.
- **Respaldo local (`LocalTextPredictor`):** Cuando el servidor de MLflow no está disponible en el entorno de nube (`No module named 'mlflow'`), `src/predict.py` carga automáticamente desde `/app/models/` el modelo `logistic_regression_airflow.joblib` y el vectorizador `tfidf_vectorizer_airflow.joblib` para transformar el texto y generar la predicción[cite: 7, 8].
- **Puerto dinámico:** Uvicorn escucha en `0.0.0.0` usando el puerto configurado por el entorno (`${PORT:-8000}`, detectado en el puerto `10000` en Render)[cite: 6, 8].

### Prueba rápida del endpoint `/predict`

```bash
curl -X POST "[https://marvel-mlops.onrender.com/predict](https://marvel-mlops.onrender.com/predict)" \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "I am Iron Man"}'
## Despliegue en producción (Render)

- **URL pública:** `https://marvel-mlops.onrender.com`
- **Documentación interactiva (Swagger UI):** `https://marvel-mlops.onrender.com/docs`
- **Endpoint de predicción:** `POST https://marvel-mlops.onrender.com/predict`
- **Respaldo en nube:** Si MLflow no está disponible, `src/predict.py` utiliza `LocalTextPredictor` con `models/logistic_regression_airflow.joblib` y `models/tfidf_vectorizer_airflow.joblib` empaquetados vía `Dockerfile.api`[cite: 7, 8].
