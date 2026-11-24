
import argparse
import hashlib
import time
from pathlib import Path

import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =====================================================
# 1. CONFIGURACIÓN GENERAL
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "mcu.csv"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"

mlflow.set_tracking_uri(DATABASE_URI)
mlflow.set_experiment("MARVEL_MLOPS_DVC_Comparison")


# =====================================================
# 2. CONFIGURACIÓN DEL DATASET
# =====================================================

MIN_DIALOGUES = 50
TEST_SIZE = 0.20
RANDOM_STATE = 42


# =====================================================
# 3. IDENTIFICAR LA VERSIÓN DEL DATASET
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "version",
    choices=["v1", "v2"],
    help="Versión del dataset que se evaluará"
)

args = parser.parse_args()

version = args.version

print("=" * 65)
print(f" COMPARACIÓN DVC - DATASET {version.upper()}")
print("=" * 65)


# =====================================================
# 4. IDENTIFICAR EL ARCHIVO UTILIZADO
# =====================================================

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {RAW_PATH}"
    )

dataset_hash = hashlib.sha256(
    RAW_PATH.read_bytes()
).hexdigest()


# =====================================================
# 5. PREPARACIÓN DE LOS DATOS
# =====================================================

df = pd.read_csv(RAW_PATH)

print("\nRegistros originales:", len(df))

df["line"] = (
    df["line"].fillna("").astype(str).str.strip()
)

df["character"] = (
    df["character"].fillna("").astype(str).str.strip()
)

df = df[
    (df["line"] != "") &
    (df["character"] != "")
].copy()

character_counts = df["character"].value_counts()

selected_characters = character_counts[
    character_counts >= MIN_DIALOGUES
].index

df = df[
    df["character"].isin(selected_characters)
].copy()

print("Personajes seleccionados:", len(selected_characters))
print("Diálogos después del filtrado:", len(df))


# =====================================================
# 6. DIVISIÓN DE ENTRENAMIENTO Y PRUEBA
# =====================================================

X = df["line"].astype(str)
y = df["character"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nDatos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))


# =====================================================
# 7. TRANSFORMACIÓN TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words="english",
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nMatriz TF-IDF:", X_train_tfidf.shape)


# =====================================================
# 8. ENTRENAMIENTO DEL MODELO SVM
# =====================================================

model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

print("\nEntrenando LinearSVC...")

start_time = time.perf_counter()

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

training_time = time.perf_counter() - start_time


# =====================================================
# 9. EVALUACIÓN DEL MODELO
# =====================================================

metrics = {
    "accuracy": accuracy_score(y_test, predictions),

    "precision_macro": precision_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    ),

    "recall_macro": recall_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    ),

    "f1_macro": f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    ),

    "f1_weighted": f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    ),

    "training_time_seconds": training_time
}


# =====================================================
# 10. GUARDAR REPORTE
# =====================================================

report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

report_path = REPORTS_DIR / f"svm_dvc_{version}_report.txt"

report_path.write_text(
    report,
    encoding="utf-8"
)


# =====================================================
# 11. REGISTRO EN MLFLOW
# =====================================================

with mlflow.start_run(
    run_name=f"SVM_DVC_{version.upper()}"
):

    mlflow.log_params({
        "model": "LinearSVC",
        "dataset_version": version,
        "dataset_sha256": dataset_hash,
        "min_dialogues": MIN_DIALOGUES,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "max_features": 20000,
        "ngram_range": "(1, 2)",
        "min_df": 2,
        "stop_words": "english",
        "class_weight": "balanced",
        "num_classes": len(selected_characters),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })

    mlflow.log_metrics({
        key: float(value)
        for key, value in metrics.items()
    })

    mlflow.set_tag("project", "MARVEL_MLOPS")
    mlflow.set_tag("comparison", "DVC dataset versions")

    mlflow.log_artifact(
        str(report_path),
        artifact_path="reports"
    )


# =====================================================
# 12. MOSTRAR RESULTADOS
# =====================================================

print("\n" + "=" * 65)
print(f" RESULTADOS DATASET {version.upper()}")
print("=" * 65)

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nReporte guardado en:", report_path)
print("Experimento registrado en MLflow.")
print("\nEVALUACIÓN FINALIZADA.")
