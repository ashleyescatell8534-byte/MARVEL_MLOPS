
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw")

files = [
    "mcu.csv",
    "mcu_subset.csv",
    "characters.csv",
    "movies.csv"
]

print("=" * 65)
print("       MARVEL MLOPS - ANALISIS EXPLORATORIO")
print("=" * 65)

for filename in files:

    filepath = DATA_PATH / filename

    print("\n" + "=" * 65)
    print(f"ARCHIVO: {filename}")
    print("=" * 65)

    if not filepath.exists():
        print("Archivo no encontrado.")
        continue

    try:
        try:
            df = pd.read_csv(filepath)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="latin-1")

        print("\nDimensiones:")
        print(f"Filas: {df.shape[0]}")
        print(f"Columnas: {df.shape[1]}")

        print("\nNombres de columnas:")
        print(df.columns.tolist())

        print("\nTipos de datos:")
        print(df.dtypes)

        print("\nPrimeros 5 registros:")
        print(df.head().to_string())

        print("\nValores faltantes:")
        print(df.isnull().sum())

        print("\nRegistros duplicados:")
        print(df.duplicated().sum())

        print("\n" + "-" * 65)

    except Exception as error:
        print(f"Error al leer el archivo: {error}")

print("\nANALISIS FINALIZADO")
