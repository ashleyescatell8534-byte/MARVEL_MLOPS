
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Rutas
RAW_PATH = Path("data/raw/mcu.csv")
PROCESSED_PATH = Path("data/processed")

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Configuración
MIN_DIALOGUES = 50
TEST_SIZE = 0.20
RANDOM_STATE = 42

print("=" * 60)
print("       MARVEL MLOPS - PREPARACION DE DATOS")
print("=" * 60)

# 1. Cargar dataset original
df = pd.read_csv(RAW_PATH)

print("\nRegistros originales:", len(df))

# 2. Limpiar diálogos vacíos
df["line"] = df["line"].fillna("").astype(str).str.strip()
df["character"] = df["character"].fillna("").astype(str).str.strip()

df = df[
    (df["line"] != "") &
    (df["character"] != "")
].copy()

# 3. Seleccionar personajes con al menos 50 diálogos
character_counts = df["character"].value_counts()

selected_characters = character_counts[
    character_counts >= MIN_DIALOGUES
].index

df = df[df["character"].isin(selected_characters)].copy()

print("Personajes seleccionados:", len(selected_characters))
print("Diálogos después del filtrado:", len(df))

# 4. Conservar únicamente las columnas necesarias
df = df[["line", "character", "movie", "year"]]

# 5. Dividir en entrenamiento y prueba
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["character"]
)

# 6. Guardar archivos
train_df.to_csv(
    PROCESSED_PATH / "train.csv",
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    PROCESSED_PATH / "test.csv",
    index=False,
    encoding="utf-8"
)

# 7. Mostrar resultados
print("\nDistribución final:")
print("Datos de entrenamiento:", len(train_df))
print("Datos de prueba:", len(test_df))
print("Total:", len(train_df) + len(test_df))

print("\nPersonajes en entrenamiento:", train_df["character"].nunique())
print("Personajes en prueba:", test_df["character"].nunique())

print("\nLos 10 personajes con más diálogos:")
print(df["character"].value_counts().head(10))

print("\nArchivos guardados:")
print("- data/processed/train.csv")
print("- data/processed/test.csv")

print("\nPreparación finalizada correctamente.")
