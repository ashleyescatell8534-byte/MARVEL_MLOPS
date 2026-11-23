
from pathlib import Path

import pandas as pd
import mlflow


# =====================================================
# 1. CONFIGURACIÓN GENERAL
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

DATABASE_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"

mlflow.set_tracking_uri(DATABASE_URI)
mlflow.set_experiment("MARVEL_MLOPS")


# =====================================================
# 2. CONFIGURACIÓN DE MODELOS CLÁSICOS
# =====================================================

CLASSICAL_MODELS = {
    "Logistic Regression": {
        "model_file": "logistic_regression.joblib",
        "algorithm": "LogisticRegression",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    },

    "Random Forest": {
        "model_file": "random_forest.joblib",
        "algorithm": "RandomForestClassifier",
        "n_estimators": 100,
        "rf_max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1,
    },

    "SVM": {
        "model_file": "svm.joblib",
        "algorithm": "LinearSVC",
        "class_weight": "balanced",
        "random_state": 42,
    },
}


# =====================================================
# 3. FUNCIONES AUXILIARES
# =====================================================

def experiment_run_exists(run_name):

    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("MARVEL_MLOPS")

    if experiment is None:
        return False

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )

    return len(runs) > 0


# =====================================================
# 4. REGISTRO DE MODELOS CLÁSICOS
# =====================================================

def log_classical_models():

    csv_path = REPORTS_DIR / "classical_models_comparison.csv"
    results = pd.read_csv(csv_path)

    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"

    for _, row in results.iterrows():

        model_name = row["model"]
        config = CLASSICAL_MODELS[model_name]

        run_name = model_name.replace(" ", "_")

        if experiment_run_exists(run_name):
            print(f"Ya existe el experimento: {run_name}")
            continue

        model_path = MODELS_DIR / config["model_file"]

        print(f"\nRegistrando: {model_name}")

        with mlflow.start_run(run_name=run_name):

            # Parámetros del algoritmo
            mlflow.log_params(config)

            # Parámetros del vectorizador TF-IDF
            mlflow.log_params({
                "vectorizer": "TF-IDF",
                "tfidf_max_features": 20000,
                "ngram_range": "(1, 2)",
                "min_df": 2,
                "stop_words": "english",
                "sublinear_tf": True,
                "dataset_classes": 68,
            })

            # Métricas
            metrics = {
                key: float(row[key])
                for key in [
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                    "f1_weighted",
                    "training_time_seconds",
                ]
            }

            mlflow.log_metrics(metrics)

            # Artefactos
            mlflow.log_artifact(
                str(model_path),
                artifact_path="model"
            )

            mlflow.log_artifact(
                str(vectorizer_path),
                artifact_path="model"
            )

            # Reporte individual
            report_path = (
                REPORTS_DIR /
                f"{config['model_file'].replace('.joblib', '_report.txt')}"
            )

            if report_path.exists():
                mlflow.log_artifact(
                    str(report_path),
                    artifact_path="reports"
                )

            print(f"Registro completado: {model_name}")


# =====================================================
# 5. REGISTRO DE BERT PONDERADO
# =====================================================

def log_bert_weighted():

    run_name = "BERT_Weighted"

    if experiment_run_exists(run_name):
        print(f"Ya existe el experimento: {run_name}")
        return

    csv_path = REPORTS_DIR / "bert_results.csv"
    results = pd.read_csv(csv_path)

    row = results.iloc[0]

    bert_dir = MODELS_DIR / "bert"

    print("\nRegistrando: BERT ponderado")

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "model": "prajjwal1/bert-tiny",
            "max_length": 64,
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 2e-5,
            "seed": 42,
            "device": "CPU",
            "class_weighted": True,
            "num_classes": 68,
        })

        metrics = {
            key: float(row[key])
            for key in [
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "training_time_seconds",
            ]
        }

        mlflow.log_metrics(metrics)

        files_to_log = [
            "config.json",
            "model.safetensors",
            "labels.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.txt",
        ]

        for filename in files_to_log:

            file_path = bert_dir / filename

            if file_path.exists():
                mlflow.log_artifact(
                    str(file_path),
                    artifact_path="bert_model"
                )

        report_path = REPORTS_DIR / "bert_report.txt"

        if report_path.exists():
            mlflow.log_artifact(
                str(report_path),
                artifact_path="reports"
            )

        print("Registro completado: BERT ponderado")


# =====================================================
# 6. REGISTRO HISTÓRICO DE BERT ORIGINAL
# =====================================================

def log_bert_original():

    run_name = "BERT_Original_Historical"

    if experiment_run_exists(run_name):
        print(f"Ya existe el experimento: {run_name}")
        return

    print("\nRegistrando: BERT original histórico")

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "model": "prajjwal1/bert-tiny",
            "max_length": 64,
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 2e-5,
            "seed": 42,
            "device": "CPU",
            "class_weighted": False,
            "num_classes": 68,
            "record_type": "historical_metrics_only",
        })

        mlflow.log_metrics({
            "accuracy": 0.1428,
            "precision_macro": 0.0021,
            "recall_macro": 0.0147,
            "f1_macro": 0.0037,
            "f1_weighted": 0.0357,
            "training_time_seconds": 1865.95,
        })

        mlflow.set_tag(
            "artifact_status",
            "Original model artifact not preserved"
        )

        print("Registro histórico completado")


# =====================================================
# 7. EJECUCIÓN PRINCIPAL
# =====================================================

if __name__ == "__main__":

    print("====================================")
    print(" INICIANDO REGISTRO EN MLFLOW")
    print("====================================")

    log_classical_models()
    log_bert_weighted()
    log_bert_original()

    print("\n====================================")
    print(" REGISTRO FINALIZADO")
    print("====================================")

    print(f"\nBase de datos: {ROOT / 'mlflow.db'}")

