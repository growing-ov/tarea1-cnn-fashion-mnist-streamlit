"""
Entrenamiento de la CNN que classify las prendas de Fashion-MNIST.

    python src/train.py
    python src/train.py --epochs 30 --batch-size 128
    python src/train.py --output models/fashion_mnist_cnn.keras

Cada paso del proceso:
  1. Bajar Fashion-MNIST desde `keras.datasets` (sin claves ni cuentas).
  2. Dejar los píxeles en [0, 1] y repartir los datos en entrenamiento/validación/prueba.
  3. Montar la red, con el aumento de datos integrado como capas de Keras.
  4. Entrenar con `EarlyStopping` y `ReduceLROnPlateau`, vigilando `val_loss`.
  5. Medir sobre el conjunto de prueba: accuracy, precision, recall, F1 y reporte por clase.
  6. Escribir el modelo (`.keras`), el histórico y las métricas en `models/` y `reports/`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # menos ruido en consola

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

# Etiquetas de Fashion-MNIST traducidas al español, en el orden 0-9 del dataset
CLASS_NAMES = [
    "Camiseta/top",
    "Pantalón",
    "Suéter",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla deportiva",
    "Bolso",
    "Botín",
]

IMG_SIZE = 28
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "fashion_mnist_cnn.keras"


def set_seed(seed: int = SEED) -> None:
    """Fija las semillas de Python, NumPy y TensorFlow (reproducibilidad aproximada en CPU)."""
    keras.utils.set_random_seed(seed)


def load_data(val_size: int = 6000, seed: int = SEED):
    """Baja Fashion-MNIST, normaliza y separa los tres conjuntos.

    - El dataset original trae 60.000 imágenes de entrenamiento y 10.000 de prueba.
    - `val_size` imágenes salen del entrenamiento (un reparto estratificado por clase) y
      se reservan para validar.
    - Los píxeles pasan a float32 en [0, 1] y se les añade el canal (28, 28, 1).
    """
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Píxeles en [0, 1] y canal explícito
    x_train_full = (x_train_full.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    # Se aparta el conjunto de validación, equilibrando clases
    rng = np.random.default_rng(seed)
    val_idx = []
    per_class = val_size // NUM_CLASSES
    for c in range(NUM_CLASSES):
        idx_c = np.where(y_train_full == c)[0]
        val_idx.extend(rng.choice(idx_c, size=per_class, replace=False))
    val_idx = np.array(sorted(val_idx))
    mask = np.ones(len(y_train_full), dtype=bool)
    mask[val_idx] = False

    x_train, y_train = x_train_full[mask], y_train_full[mask]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_augmentation() -> keras.Sequential:
    """Aumento de datos: solo entra en juego durante el entrenamiento (training=True)."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomZoom(0.08),
        ],
        name="data_augmentation",
    )


def build_model(learning_rate: float = 1e-3) -> keras.Model:
    """Tres bloques convolucionales con BatchNorm y Dropout, y un clasificador denso."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="imagen")
    x = build_augmentation()(inputs)

    # Bloque 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Bloque 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Bloque 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Clasificador
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="probabilidades")(x)

    model = keras.Model(inputs, outputs, name="cnn_fashion_mnist")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(patience: int = 5) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]


def evaluate(model: keras.Model, x_test, y_test) -> dict:
    """Mide el modelo sobre el conjunto de prueba y devuelve accuracy, precision, recall y F1."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    y_prob = model.predict(x_test, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, digits=4
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena la CNN sobre Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    set_seed()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    print(f"Entrenamiento: {x_train.shape}, Validación: {x_val.shape}, Prueba: {x_test.shape}")

    model = build_model(args.lr)
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=get_callbacks(args.patience),
        verbose=2,
    )

    metrics = evaluate(model, x_test, y_test)
    print("\n=== Resultados en el conjunto de prueba ===")
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        print(f"{k:>16}: {metrics[k]:.4f}")
    print("\n" + metrics["classification_report"])

    # Modelo y artefactos de esta ejecución
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (reports / "history_train_py.json").write_text(json.dumps(hist, indent=2))
    (reports / "metrics_train_py.json").write_text(
        json.dumps({**metrics, "class_names": CLASS_NAMES}, indent=2, ensure_ascii=False)
    )
    print(f"\nModelo guardado en: {args.output}")


if __name__ == "__main__":
    main()
