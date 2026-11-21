# Clasificador de prendas con CNN — Fashion-MNIST

Red convolucional entrenada sobre **Fashion-MNIST** que decide a qué categoría pertenece una prenda
(Camiseta, Pantalón, Suéter, Vestido, Abrigo, Sandalia, Camisa, Zapatilla, Bolso o Botín), más una
aplicación web en Streamlit que carga ese mismo modelo y devuelve la clase predicha junto con su
confianza.

---

## Puesta en marcha

```bash
python3.12 -m venv .venv          # en Windows: python -m venv .venv
source .venv/bin/activate         # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Con el entorno listo hay tres cosas que se pueden hacer:

| Acción | Comando |
|---|---|
| Levantar la app web | `streamlit run app.py` → http://localhost:8501 |
| Predecir una imagen suelta | `python src/predict.py samples/9_botín.png` |
| Reentrenar el modelo desde cero | `python src/train.py --epochs 30 --batch-size 128` |

> El modelo ya está entrenado y versionado en `models/fashion_mnist_cnn.keras`, así que los dos
> primeros comandos funcionan sin ejecutar el entrenamiento.

---

## Qué hay dentro del repositorio

```
.
├── app.py                          # interfaz Streamlit (carga el modelo y predice)
├── notebooks/
│   └── CNN_experimentacion.ipynb   # notebook ejecutado: EDA, entrenamiento, métricas, gráficas
├── src/
│   ├── train.py                    # entrenamiento por línea de comandos
│   └── predict.py                  # predicción por línea de comandos
├── models/
│   └── fashion_mnist_cnn.keras     # modelo entrenado (1,9 MB)
├── reports/                        # history.json y metrics.json del entrenamiento
├── samples/                        # diez imágenes de ejemplo, una por clase
├── requirements.txt
└── README.md
```

---

## El recorrido técnico, resumido

**Datos.** `keras.datasets.fashion_mnist` entrega 60.000 imágenes de entrenamiento y 10.000 de
prueba, en escala de grises y de 28×28 píxeles, repartidas en 10 clases equilibradas. No hace falta
ninguna clave de API: se descargan de los servidores públicos de Google. El conjunto de
entrenamiento se vuelve a partir en **54.000 para entrenar y 6.000 para validar** (reparto
estratificado); las 10.000 de prueba no se tocan hasta la medición final.

**Preprocesamiento.** Los píxeles pasan de `uint8` a `float32` y se dividen entre 255 para quedar en
[0, 1], y se añade el canal explícito `(28, 28, 1)` que espera la primera capa convolucional.

**Arquitectura.** `Input(28,28,1)` → aumento de datos → tres bloques convolucionales
(`Conv2D` 32→64→128, `BatchNormalization`, `MaxPooling`, `Dropout`) → `GlobalAveragePooling` →
`Dense(128)` + `Dropout(0.4)` → `Dense(10, softmax)`. unos **157.000 parámetros**. Optimizador Adam
(`lr = 0.001`) y pérdida `sparse_categorical_crossentropy`.

**Aumento de datos.** Volteo horizontal, rotación de 0,05 rad (±2,9°), traslación de hasta el 8 % y
zoom de hasta el 8 %, aplicados como capas de Keras dentro del modelo, de modo que solo actúan
mientras se entrena.

**Entrenamiento.** Hasta 30 épocas con `batch_size = 128` y dos frenos: `EarlyStopping` sobre
`val_loss` (paciencia 5, restaurando los mejores pesos) y `ReduceLROnPlateau` (la tasa de
aprendizaje se reduce a la mitad tras 2 épocas sin mejora).

---

## Resultados sobre el conjunto de prueba

| Métrica | Valor |
|---|---|
| Accuracy | **0,9148** |
| Precision (macro) | 0,9180 |
| Recall (macro) | 0,9148 |
| F1 (macro) | 0,9157 |
| Loss | 0,2257 |

La mejor época fue la **28 de 30**; el entrenamiento completó en unos 17 minutos en CPU
(Apple Silicon), sin GPU.

Como las diez clases están equilibradas, los promedios macro y ponderado prácticamente coinciden, y
la accuracy describe bien el comportamiento real del modelo. Las clases que mejor se resuelven son
*Pantalón*, *Bolso*, *Sandalia*, *Botín* y *Zapatilla deportiva* (F1 > 0,95), todas con siluetas
fácilmente distinguibles. La más problemática es **Camisa (F1 ≈ 0,75)**: a 28×28 píxeles en escala
de grises es casi indistinguible de *Camiseta/top*, *Abrigo* y *Suéter*, y la matriz de confusión lo
confirma (152 casos de *Camiseta/top* clasificados como *Camisa*).

En el notebook están las curvas de entrenamiento, la matriz de confusión (conteos y normalizada),
el `classification_report` por clase, 20 predicciones con su confianza, las 10 con mayor
probabilidad de error y las imágenes de muestra que usa la app.

---

## Qué hace la app

1. Se sube una imagen (PNG/JPG) o se elige una de las diez de `samples/` en la barra lateral.
2. La app muestra la imagen original a la izquierda y, en el centro, la entrada real que recibe el
   modelo: 28×28 en escala de grises, ampliada para que se aprecie.
3. A la derecha aparecen la **clase predicha**, su **confianza** (con barra de progreso), el **top-3**
   y un gráfico con la probabilidad de las diez clases.
4. Si la confianza baja del 50 %, la app avisa de que la imagen se parece poco a lo que el modelo
   vio durante el entrenamiento.

**Preprocesamiento de imágenes externas** (en `src/predict.py`): escala de grises → inversión
automática si el fondo es claro (en Fashion-MNIST la prenda es clara sobre fondo negro) → recorte
al objeto y relleno cuadrado → redimensionado a 28×28 → normalización a [0, 1]. Las predicciones
mejoran con fotos de una sola prenda, centrada y sobre fondo uniforme.

---

## Ejecutarlo en Google Colab

1. Abre [Google Colab](https://colab.research.google.com/) → *Archivo → Subir notebook* y elige
   `notebooks/CNN_experimentacion.ipynb`. Si el repositorio está en GitHub, también funciona
   *Archivo → Abrir notebook → GitHub* pegando la URL.
2. Opcionalmente, *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU* para entrenar
   más rápido.
3. *Entorno de ejecución → Ejecutar todas*. La primera celda comprueba qué falta y el dataset se
   descarga sin credenciales.
4. El modelo entrenado queda en `models/fashion_mnist_cnn.keras` dentro del entorno de Colab; se
   descarga desde el panel de archivos si hace falta.

Para la interfaz web, la ejecución local es lo más cómodo: Colab no expone puertos web de forma
directa.

Para regenerar todas las salidas del notebook en local:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/CNN_experimentacion.ipynb
```

> El notebook y `src/train.py` sobrescriben `models/fashion_mnist_cnn.keras`. El archivo que está
> versionado en git se generó ejecutando el notebook.

---

## Limitaciones conocidas

- El modelo se entrenó con imágenes de catálogo de 28×28 muy homogéneas. Fotos reales con fondos
  complicados, varias prendas o personas en escena se clasifican mal.
- *Camisa*, *Camiseta/top*, *Suéter* y *Abrigo* comparten silueta a esta resolución; es el techo del
  modelo en este dataset.
- Parte del error residual parece venir de etiquetas dudosas del propio conjunto, no solo del
  modelo: hay prendas etiquetadas como *Pantalón* que en realidad parecen un *Vestido*.
