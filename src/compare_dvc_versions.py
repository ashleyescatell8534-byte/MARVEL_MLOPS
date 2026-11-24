
import sys
import hashlib
import time
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------------------------
# 1. CONFIGURACIÓN
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mcu.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MIN_DIALOGUES = 50

# Conexión a la base de datos de MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "MARVEL_MLOPS_DVC_Comparison"
mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# 2. IDENTIFICAR LA VERSIÓN DEL DATASET
# --------------------------------------------------

if len(sys.argv) < 2 or sys.argv[1].lower() not in ["v1", "v2"]:
    print("Uso: python src/compare_dvc_versions.py v1")
    print("O:   python src/compare_dvc_versions.py v2")
    sys.exit(1)

version = sys.argv[1].lower()

print("\n" + "=" * 55)
print(f"COMPARACIÓN DVC - DATASET {version.upper()}")
print("=" * 55)


# --------------------------------------------------
# 3. CARGAR Y VERIFICAR EL DATASET
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

# Calcular hash SHA256 para identificar exactamente
# la versión del archivo utilizada
sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

print(f"Registros originales: {len(df)}")

# Verificar las columnas necesarias
required_columns = ["character", "line"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Falta la columna requerida: {column}"
        )

# Eliminar registros sin personaje o diálogo
df = df[["character", "line"]].dropna()

df["character"] = df["character"].astype(str)
df["line"] = df["line"].astype(str)

df = df[
    (df["character"].str.strip() != "") &
    (df["line"].str.strip() != "")
]


# --------------------------------------------------
# 4. FILTRAR PERSONAJES
# --------------------------------------------------

# Seleccionar personajes con al menos 50 diálogos
character_counts = df["character"].value_counts()

selected_characters = character_counts[
    character_counts >= MIN_DIALOGUES
].index

df = df[df["character"].isin(selected_characters)]

print(f"Personajes seleccionados: {len(selected_characters)}")
print(f"Diálogos después del filtrado: {len(df)}")

X = df["line"]
y = df["character"]


# --------------------------------------------------
# 5. DIVIDIR LOS DATOS
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Datos de entrenamiento: {len(X_train)}")
print(f"Datos de prueba: {len(X_test)}")


# --------------------------------------------------
# 6. TRANSFORMAR TEXTO A TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words="english",
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Matriz TF-IDF: {X_train_tfidf.shape}")


# --------------------------------------------------
# 7. ENTRENAR MODELO SVM
# --------------------------------------------------

model = LinearSVC(
    class_weight="balanced",
    random_state=RANDOM_STATE
)

print("\nEntrenando LinearSVC...")

start_time = time.time()

model.fit(X_train_tfidf, y_train)

training_time = time.time() - start_time


# --------------------------------------------------
# 8. REALIZAR PREDICCIONES
# --------------------------------------------------

y_pred = model.predict(X_test_tfidf)

# Obtener puntuaciones de decisión para calcular ROC-AUC
decision_scores = model.decision_function(X_test_tfidf)

# Convertir las etiquetas reales a formato binario
# para calcular ROC-AUC One-vs-Rest multiclase
y_test_bin = label_binarize(
    y_test,
    classes=model.classes_
)


# --------------------------------------------------
# 9. CALCULAR MÉTRICAS
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision_macro = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# ROC-AUC macro para clasificación multiclase
roc_auc_macro = roc_auc_score(
    y_test_bin,
    decision_scores,
    average="macro"
)

metrics = {
    "accuracy": accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "f1_weighted": f1_weighted,
    "roc_auc_macro": roc_auc_macro,
    "training_time_seconds": training_time
}


# --------------------------------------------------
# 10. MOSTRAR RESULTADOS
# --------------------------------------------------

print("\n" + "=" * 55)
print(f"RESULTADOS DATASET {version.upper()}")
print("=" * 55)

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")


# --------------------------------------------------
# 11. REGISTRAR EXPERIMENTO EN MLFLOW
# --------------------------------------------------

run_name = f"SVM_DVC_{version.upper()}_ROC_AUC"

with mlflow.start_run(run_name=run_name):

    mlflow.log_param("dataset_version", version)
    mlflow.log_param("dataset_sha256", sha256)
    mlflow.log_param("model", "LinearSVC")
    mlflow.log_param("minimum_dialogues", MIN_DIALOGUES)
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("max_features", 20000)
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("number_of_characters", len(selected_characters))

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    print("\nExperimento registrado en MLflow.")


# --------------------------------------------------
# 12. GUARDAR REPORTE
# --------------------------------------------------

report_path = REPORTS_DIR / f"svm_dvc_{version}_report.txt"

with open(report_path, "w", encoding="utf-8") as report:

    report.write(f"COMPARACIÓN DVC - DATASET {version.upper()}\n")
    report.write("=" * 45 + "\n\n")

    report.write(f"Registros originales: {len(pd.read_csv(DATA_PATH))}\n")
    report.write(f"Personajes seleccionados: {len(selected_characters)}\n")
    report.write(f"Diálogos después del filtrado: {len(df)}\n")
    report.write(f"Datos de entrenamiento: {len(X_train)}\n")
    report.write(f"Datos de prueba: {len(X_test)}\n")
    report.write(f"SHA256: {sha256}\n\n")

    report.write("MÉTRICAS DEL MODELO\n")
    report.write("-" * 30 + "\n")

    for metric_name, metric_value in metrics.items():
        report.write(f"{metric_name}: {metric_value:.6f}\n")

print(f"\nReporte guardado en: {report_path}")

print("\nProceso terminado correctamente.")
