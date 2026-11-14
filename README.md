# Red Neuronal Convolucional (CNN) + Streamlit — Fashion-MNIST

Proyecto de clasificación de imágenes con una **red neuronal convolucional** (Keras/TensorFlow) entrenada sobre **Fashion-MNIST**, con *Data Augmentation*, *EarlyStopping*, evaluación completa y una **aplicación web en Streamlit** que carga el modelo entrenado y predice la clase de una imagen subida por el usuario.

## Dataset

**Fashion-MNIST** (Zalando Research), descargado automáticamente con `keras.datasets.fashion_mnist` (sin autenticación):

- 70.000 imágenes en escala de grises de 28×28 píxeles (60.000 entrenamiento + 10.000 prueba).
- 10 clases balanceadas: Camiseta/top, Pantalón, Suéter, Vestido, Abrigo, Sandalia, Camisa, Zapatilla deportiva, Bolso, Botín.
- Se eligió frente a CIFAR-10 porque entrena en pocos minutos en CPU / Apple Silicon y aun así presenta clases difíciles de separar (camisa vs. camiseta vs. abrigo vs. suéter).

División usada: **54.000 entrenamiento / 6.000 validación** (estratificada, a partir del conjunto de entrenamiento original) y **10.000 prueba**.

## Resultados (conjunto de prueba)

| Métrica | Valor |
|---|---|
| Accuracy | **0,9148** |
| Precision (macro) | 0,9180 |
| Recall (macro) | 0,9148 |
| F1 (macro) | 0,9157 |
| Loss | 0,2257 |

- Mejor época: 28 de 30 (EarlyStopping con `restore_best_weights=True`); ~17 min de entrenamiento en CPU (Apple Silicon).
- Clases más fáciles: Pantalón, Bolso, Sandalia (F1 ≈ 0,98). Más difícil: Camisa (F1 ≈ 0,75), confundida sobre todo con Camiseta/top, Abrigo y Suéter.

Las gráficas de entrenamiento, la matriz de confusión, el `classification_report` y más de 30 predicciones visualizadas están en el notebook ejecutado [`notebooks/CNN_experimentacion.ipynb`](notebooks/CNN_experimentacion.ipynb).

## Estructura del repositorio

```
.
├── app.py                          # App Streamlit: carga imagen → clase, confianza y top-3
├── notebooks/
│   └── CNN_experimentacion.ipynb   # Exploración, entrenamiento, evaluación (ejecutado, con salidas)
├── src/
│   ├── train.py                    # Entrenamiento por línea de comandos
│   └── predict.py                  # Predicción por línea de comandos (y funciones usadas por la app)
├── models/
│   └── fashion_mnist_cnn.keras     # Modelo entrenado (versionado en git)
├── reports/                        # Historial de entrenamiento y métricas (JSON)
├── samples/                        # Imágenes de ejemplo (una por clase) para probar
├── requirements.txt
└── README.md
```

## Arquitectura de la CNN

`Input 28×28×1` → **Data Augmentation** (flip horizontal, rotación ±5 %, traslación ±8 %, zoom ±8 %) →
3 bloques convolucionales (`Conv2D` 32→64→128 + `BatchNormalization` + `MaxPooling` + `Dropout`) →
`GlobalAveragePooling` → `Dense(128)` + `Dropout(0.4)` → `Dense(10, softmax)`.
≈157 mil parámetros. Optimizador Adam, pérdida `sparse_categorical_crossentropy`,
callbacks `EarlyStopping(val_loss, patience=5, restore_best_weights=True)` y `ReduceLROnPlateau`.

## Ejecución local

Requiere **Python 3.10–3.12** (TensorFlow aún no publica ruedas para versiones más nuevas en todas las plataformas).

```bash
# 1. Crear y activar el entorno virtual
python3.12 -m venv .venv            # o: uv venv .venv --python 3.12
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (Opcional) Reentrenar el modelo — ya hay uno entrenado en models/
python src/train.py                 # opciones: --epochs 30 --batch-size 128 --lr 0.001

# 4. Predecir sobre una imagen
python src/predict.py samples/9_botín.png
python src/predict.py mi_foto.jpg --top 3

# 5. Lanzar la aplicación web
streamlit run app.py                # abre http://localhost:8501
```

Para volver a ejecutar el notebook completo y regenerar todas las salidas:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/CNN_experimentacion.ipynb
```

> El notebook y `src/train.py` sobrescriben `models/fashion_mnist_cnn.keras`. El modelo versionado se generó ejecutando el notebook.

## Ejecución en Google Colab

1. Abre [Google Colab](https://colab.research.google.com/) → *Archivo → Subir notebook* y selecciona `notebooks/CNN_experimentacion.ipynb`
   (o, si el repositorio está en GitHub: *Archivo → Abrir notebook → GitHub* y pega la URL).
2. (Opcional) *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU* para entrenar más rápido.
3. *Entorno de ejecución → Ejecutar todas*. La primera celda verifica/instala dependencias y el dataset se descarga desde `keras.datasets` sin autenticación.
4. El modelo se guarda en `models/fashion_mnist_cnn.keras` dentro del entorno de Colab; descárgalo desde el panel de archivos si lo necesitas.

Para la app Streamlit se recomienda ejecución local (Colab no expone puertos web directamente).

## Uso de la aplicación

1. Ejecuta `streamlit run app.py`.
2. Sube una imagen (PNG/JPG) de una prenda o calzado, o elige un ejemplo de `samples/` en la barra lateral.
3. La app muestra la imagen original, la entrada preprocesada de 28×28, la **clase predicha** con su **confianza**, el **top-3** y un gráfico con la probabilidad de cada clase.

**Preprocesamiento de imágenes externas** (`src/predict.py`): escala de grises → inversión automática si el fondo es claro (Fashion-MNIST tiene la prenda clara sobre fondo negro) → recorte al objeto y relleno cuadrado → redimensionado a 28×28 → normalización a [0, 1]. Las mejores predicciones se obtienen con fotos de una sola prenda, centrada y sobre fondo uniforme.

## Limitaciones

- El modelo aprendió de imágenes de catálogo 28×28 muy homogéneas; fotos reales con fondos complejos, varias prendas o personas pueden clasificarse mal.
- Las clases Camisa, Camiseta/top, Suéter y Abrigo son las más confundidas por su similitud visual a baja resolución.
