
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