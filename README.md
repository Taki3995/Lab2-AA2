# Laboratorio 2: Clasificación de Sentimientos con Transformers

Este proyecto es una implementación de un pipeline de Machine Learning para la **clasificación binaria de sentimientos** (positivo/negativo) de reseñas de películas del dataset IMDB.

El objetivo principal es construir los componentes clave de un modelo Transformer desde cero, incluyendo un tokenizador BPE (Byte-Pair Encoding) personalizado y el mecanismo de Multi-Head Attention, para luego entrenarlo y evaluar su rendimiento.

El modelo final alcanza una **precisión de ~89.7%** en el conjunto de pruebas.

## Características Principales

* **Análisis Exploratorio de Datos (EDA):** Visualización de la distribución de longitudes de reseñas y balance de clases.
* **Tokenizador BPE Personalizado:** Implementación de un tokenizador Byte-Pair Encoding desde cero, que construye un vocabulario de 30,000 sub-palabras a partir del corpus de IMDB.
* **Atención Multi-Cabeza (Multi-Head Attention):** Implementación personalizada de las capas `ScaledDotProductAttention` y `MultiHeadAttention`.
* **Arquitectura Transformer Encoder:** Construcción de un modelo apilando múltiples capas de `TransformerEncoderLayer` (incluyendo conexiones residuales, LayerNorm y FFN).
* **Entrenamiento y Evaluación:**
    * Uso del optimizador `AdamW` y un scheduler `ReduceLROnPlateau` para un entrenamiento estable.
    * Implementación de **Early Stopping** para prevenir el sobreajuste.
    * Análisis de rendimiento detallado en el conjunto de prueba, incluyendo una matriz de confusión y un desglose de la precisión por longitud de reseña.

## Cómo Empezar

### Prerrequisitos

* Python 3.8+
* PyTorch
* Jupyter Notebook, Jupyter Lab o Google Colab

### Instalación

1.  Clona este repositorio (o descarga los archivos):
    ```bash
    git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
    cd tu-repositorio
    ```

2.  Crea un entorno virtual (recomendado):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Windows: .venv\Scripts\activate
    ```

3.  Instala las dependencias necesarias:
    ```bash
    pip install torch datasets numpy scikit-learn matplotlib seaborn
    ```

### 🏃 Ejecución

1.  Abre el Jupyter Notebook:
    ```bash
    jupyter lab "Lab2 2025.ipynb"
    ```

2.  **Importante! Activa la GPU:**
    * **En Google Colab:** Ve a `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` y selecciona `T4 GPU` como acelerador de hardware.
    * **Localmente:** Asegúrate de tener una GPU compatible con CUDA y que PyTorch la esté detectando.

3.  Ejecuta todas las celdas en orden. El script hará lo siguiente:
    * Cargará y analizará los datos.
    * **Construirá el vocabulario BPE de 30,000 tokens** (esto puede tardar varios minutos).
    * Definirá la arquitectura del modelo.
    * Entrenará el modelo, guardando el mejor checkpoint como `best_model.pth`.
    * Evaluará el modelo en el conjunto de prueba.
    * Ejecutará pruebas con ejemplos personalizados.

## 🧠 Detalles de Implementación

### 1. Tokenizador BPE (`SimpleTokenizer`)

Se implementó un tokenizador BPE desde cero. En lugar de usar un tokenizador pre-entrenado, este:
1.  **Limpia** los textos y cuenta la frecuencia de las palabras (añadiendo `</w>` al final).
2.  **Inicializa** el vocabulario con todos los caracteres base.
3.  **Iterativamente** cuenta los pares de símbolos más frecuentes y los fusiona.
4.  **Aprende** 30,000 "merges" (fusiones) para crear un vocabulario de sub-palabras.
5.  Proporciona métodos `encode()` y `decode()` que utilizan estos "merges" aprendidos.

### 2. Modelo Transformer

El modelo (`SentimentAnalysisModel`) no utiliza `nn.TransformerEncoder`, sino que construye la pila manualmente:

* **Embedding + Positional Encoding:** Convierte los IDs de los tokens en vectores y les suma la información posicional.
* **Pila de N Capas Encoder:** El modelo utiliza **4 capas** (`num_layers = 4`) de `TransformerEncoderLayer`.
* **Capa Encoder Personalizada:** Cada `TransformerEncoderLayer` contiene:
    1.  Una capa `MultiHeadAttention` personalizada (**8 cabezas**, `num_heads = 8`).
    2.  Una conexión residual y `LayerNorm`.
    3.  Una red FeedForward (FFN).
    4.  Otra conexión residual y `LayerNorm`.
* **Clasificación:** La salida de la secuencia del Transformer (shape `[Batch, SeqLen, DimModel]`) se promedia a lo largo de la dimensión de la secuencia (`.mean(dim=1)`) para obtener un vector de sentimiento único por reseña, que luego se pasa a una capa lineal final para la clasificación.

## Resultados

Tras el entrenamiento con *Early Stopping*, el modelo detuvo su entrenamiento en la **Época 6** al no encontrar mejoras en la pérdida de validación.

* **Precisión de Prueba (Test Accuracy):** **89.66%**
* **Mejor Pérdida de Validación:** 0.3076 (alcanzada en la Época 1)

### Matriz de Confusión (Prueba)

| | Pred. Negativo | Pred. Positivo |
| :--- | :---: | :---: |
| **Real Negativo**| 2235 | 277 |
| **Real Positivo**| 240 | 2248 |

### Análisis por Longitud de Reseña

El modelo demostró ser robusto independientemente de la longitud del texto, gracias al `max_length=512` y al mecanismo de atención.

* **Reseñas Cortas (<150 palabras):** 89.86%
* **Reseñas Medianas (150-300 palabras):** 89.33%
* **Reseñas Largas (>=300 palabras):** 89.88%

### Prueba en Español (Out-of-Distribution)

Como era de esperar, el modelo falla al clasificar texto en español, ya que su vocabulario BPE se construyó exclusivamente con el corpus en inglés de IMDB.

* **Texto:** `La pelicula fue bastante aburrida, carecía de buena trama`
* **Predicción:** **Positive** (Incorrecta)

Esto demuestra que el vocabulario es el componente fundamental que limita el modelo a un idioma específico.
