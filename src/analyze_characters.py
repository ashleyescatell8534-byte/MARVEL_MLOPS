
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/mcu.csv")

df = pd.read_csv(DATA_PATH)

# Contar cuántos diálogos tiene cada personaje
character_counts = df["character"].value_counts()

print("=" * 60)
print("       DISTRIBUCION DE DIALOGOS POR PERSONAJE")
print("=" * 60)

print("\nTotal de personajes:", df["character"].nunique())
print("Total de dialogos:", len(df))

print("\nCantidad de personajes según número de diálogos:")
print("Con 1 diálogo:", (character_counts == 1).sum())
print("Con menos de 10:", (character_counts < 10).sum())
print("Con 10 o más:", (character_counts >= 10).sum())
print("Con 20 o más:", (character_counts >= 20).sum())
print("Con 50 o más:", (character_counts >= 50).sum())

print("\nLos 30 personajes con más diálogos:")
print(character_counts.head(30).to_string())

print("\nDistribución de cantidad de diálogos:")
print(character_counts.describe())

# Guardar el conteo para consultarlo después
character_counts.to_csv(
    "reports/character_distribution.csv",
    header=["dialogue_count"]
)

print("\nAnálisis terminado.")
