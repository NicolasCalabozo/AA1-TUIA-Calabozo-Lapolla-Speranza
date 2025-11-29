# IA 4.1 Aprendizaje Automático I - Trabajo Práctico N°2
 Facultad de Ciencias Exactas, Ingeniería y Agrimensura
 
 Tecnicatura Universitaria en Inteligencia Artificial

## Integrantes

* Calabozo, Nicolás
* Lapolla, Martín
* Speranza, Emanuel

Este repositorio contiene la resolución del Trabajo Práctico N°2 de Aprendizaje Automático I, el cual implementa un modelo de **Red Neuronal** optimizado con **Optuna** para predecir si lloverá al día siguiente en Australia (`RainTomorrow`).

## Requisitos Previos

* **Docker Desktop** (para Windows/Mac) o Docker Engine (Linux).
* *Windows:* Es necesario que Docker Desktop esté abierto y ejecutándose.

## Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/NicolasCalabozo/AA1-TUIA-Calabozo-Lapolla-Speranza.git
    cd AA1-TUIA-Calabozo-Lapolla-Speranza
    ```

## Instrucciones de Uso

1.  **Crear contenedor de Docker:**
    ```bash
    docker build -t "nombre del contenedor" .
    ```
2.  **Ejecutar el contenedor:**
    ```bash
    docker run --rm -v ${PWD}:/app "nombre del contenedor" python inference.py
    ```