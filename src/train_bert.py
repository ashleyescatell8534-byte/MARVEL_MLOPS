

import time
import json
import gc
import pandas as pd
import numpy as np
import torch

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

# =====================================================
# CONFIGURACION
# =====================================================

DATA_PATH = Path("data/processed")
MODEL_PATH = Path("models/bert")
REPORT_PATH = Path("reports")

MODEL_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LENGTH = 64
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 65)
print("       MARVEL MLOPS - ENTRENAMIENTO BERT")
print("=" * 65)

# =====================================================
# 1. CARGAR DATOS
# =====================================================

train_df = pd.read_csv(DATA_PATH / "train.csv")
test_df = pd.read_csv(DATA_PATH / "test.csv")

print("\nDatos cargados correctamente.")
print("Entrenamiento:", len(train_df))
print("Prueba:", len(test_df))

label_encoder = LabelEncoder()

train_labels = label_encoder.fit_transform(
    train_df["character"]
)

test_labels = label_encoder.transform(
    test_df["character"]
)

num_labels = len(label_encoder.classes_)

print("Número de personajes:", num_labels)

# =====================================================
# 2. PONDERACION DE CLASES
# =====================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
)

print("\nPonderación de clases calculada.")
print("Peso mínimo:", round(class_weights.min().item(), 4))
print("Peso máximo:", round(class_weights.max().item(), 4))

# Guardar categorías

with open(
    MODEL_PATH / "labels.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        label_encoder.classes_.tolist(),
        file,
        ensure_ascii=False,
        indent=4
    )

# =====================================================
# 3. CARGAR TOKENIZER
# =====================================================

print("\nCargando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

# =====================================================
# 4. PREPARAR DATASET
# =====================================================

class MarvelDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):

        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            int(self.labels[index]),
            dtype=torch.long
        )

        return item

print("\nTokenizando diálogos...")

train_dataset = MarvelDataset(
    train_df["line"],
    train_labels,
    tokenizer,
    MAX_LENGTH
)

test_dataset = MarvelDataset(
    test_df["line"],
    test_labels,
    tokenizer,
    MAX_LENGTH
)

# =====================================================
# 5. CARGAR MODELO BERT
# =====================================================

print("\nCargando modelo BERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

# =====================================================
# 6. TRAINER CON PONDERACION DE CLASES
# =====================================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )

        loss = loss_function(
            logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

# =====================================================
# 7. CONFIGURAR ENTRENAMIENTO
# =====================================================

training_args = TrainingArguments(
    output_dir=str(MODEL_PATH / "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="no",
    report_to="none",
    use_cpu=True,
    seed=SEED
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# =====================================================
# 8. ENTRENAR
# =====================================================

print("\nIniciando entrenamiento BERT con ponderación de clases en CPU...")

start_time = time.time()

trainer.train()

training_time = time.time() - start_time

print("\nEntrenamiento terminado.")
print(f"Tiempo: {training_time:.2f} segundos")

# =====================================================
# 9. EVALUAR
# =====================================================

print("\nEvaluando modelo...")

predictions_output = trainer.predict(test_dataset)

predictions = np.argmax(
    predictions_output.predictions,
    axis=1
)

accuracy = accuracy_score(
    test_labels,
    predictions
)

precision = precision_score(
    test_labels,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    test_labels,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    test_labels,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    test_labels,
    predictions,
    average="weighted",
    zero_division=0
)

print("\nRESULTADOS BERT CON PONDERACION")
print("Accuracy:", round(accuracy, 4))
print("Precision macro:", round(precision, 4))
print("Recall macro:", round(recall, 4))
print("F1 macro:", round(f1, 4))
print("F1 weighted:", round(weighted_f1, 4))

# =====================================================
# 10. GUARDAR MODELO
# =====================================================

model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print("\nModelo guardado en:", MODEL_PATH)

# =====================================================
# 11. GUARDAR REPORTE
# =====================================================

report = classification_report(
    test_labels,
    predictions,
    labels=np.arange(num_labels),
    target_names=label_encoder.classes_,
    zero_division=0
)

with open(
    REPORT_PATH / "bert_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)

metrics = {
    "model": "BERT",
    "accuracy": accuracy,
    "precision_macro": precision,
    "recall_macro": recall,
    "f1_macro": f1,
    "f1_weighted": weighted_f1,
    "training_time_seconds": training_time
}

pd.DataFrame([metrics]).to_csv(
    REPORT_PATH / "bert_results.csv",
    index=False
)

print("\nReporte guardado en reports/bert_report.txt")
print("Métricas guardadas en reports/bert_results.csv")

# =====================================================
# 12. LIBERAR MEMORIA
# =====================================================

del trainer
del model
del train_dataset
del test_dataset

gc.collect()

print("\nENTRENAMIENTO BERT FINALIZADO.")