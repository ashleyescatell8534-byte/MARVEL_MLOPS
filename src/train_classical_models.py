import pandas as pd
import joblib
import time

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Rutas
DATA_PATH = Path("data/processed")
MODEL_PATH = Path("models")
REPORT_PATH = Path("reports")

MODEL_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("       MARVEL MLOPS - ENTRENAMIENTO DE MODELOS")
print("=" * 65)

# 1. Cargar datos
train_df = pd.read_csv(DATA_PATH / "train.csv")
test_df = pd.read_csv(DATA_PATH / "test.csv")

X_train = train_df["line"].astype(str)
y_train = train_df["character"]

X_test = test_df["line"].astype(str)
y_test = test_df["character"]

print("\nDatos cargados correctamente.")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

# 2. Convertir diálogos a vectores TF-IDF
print("\nTransformando diálogos con TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words="english",
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Matriz de entrenamiento:", X_train_tfidf.shape)
print("Matriz de prueba:", X_test_tfidf.shape)

joblib.dump(
    vectorizer,
    MODEL_PATH / "tfidf_vectorizer.joblib"
)

# 3. Definir modelos
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}

results = []

# 4. Entrenar y evaluar
for name, model in models.items():

    print("\n" + "=" * 65)
    print("Entrenando:", name)
    print("=" * 65)

    start_time = time.time()

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    elapsed_time = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision macro: {precision:.4f}")
    print(f"Recall macro: {recall:.4f}")
    print(f"F1 macro: {f1:.4f}")
    print(f"F1 weighted: {weighted_f1:.4f}")
    print(f"Tiempo: {elapsed_time:.2f} segundos")

    # Guardar modelo
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, MODEL_PATH / filename)

    # Guardar reporte por personaje
    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    report_filename = (
        name.lower().replace(" ", "_") + "_report.txt"
    )

    with open(REPORT_PATH / report_filename, "w", encoding="utf-8") as file:
        file.write(report)

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_weighted": weighted_f1,
        "training_time_seconds": elapsed_time
    })

# 5. Guardar comparación de resultados
results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_PATH / "classical_models_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("       COMPARACION FINAL DE MODELOS")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nModelos guardados en:", MODEL_PATH)
print("Reportes guardados en:", REPORT_PATH)
print("\nENTRENAMIENTO FINALIZADO.")
