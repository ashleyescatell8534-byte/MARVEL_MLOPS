
from pathlib import Path

# Ubicación de los datos originales
DATA_PATH = Path("data/raw")

print("=" * 55)
print("       MARVEL MLOPS - INSPECCION DEL DATASET")
print("=" * 55)

if not DATA_PATH.exists():
    print("ERROR: No se encontro la carpeta data/raw")
else:
    files = list(DATA_PATH.rglob("*"))

    files = [f for f in files if f.is_file()]

    if not files:
        print("No se encontraron archivos en data/raw.")
    else:
        print("\nArchivos encontrados:\n")

        for file in files:
            print(f"Nombre: {file.name}")
            print(f"Extension: {file.suffix}")
            print(f"Tamaño: {file.stat().st_size / 1024:.2f} KB")
            print("-" * 40)

        print(f"\nTotal de archivos: {len(files)}")
