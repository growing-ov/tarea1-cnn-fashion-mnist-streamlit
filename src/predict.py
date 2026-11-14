"""
Predicción con el modelo CNN entrenado (Fashion-MNIST).

Uso:
    python src/predict.py ruta/a/imagen.png
    python src/predict.py ruta/a/imagen.jpg --model models/fashion_mnist_cnn.keras --top 3

Preprocesamiento aplicado a cualquier imagen de entrada (para parecerse a Fashion-MNIST):
  1. Conversión a escala de grises.
  2. Inversión automática si el fondo es claro (Fashion-MNIST tiene prenda clara sobre fondo negro).
  3. Recorte al contenido + relleno cuadrado, y redimensionado a 28x28.
  4. Normalización a [0, 1].
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "fashion_mnist_cnn.keras"
IMG_SIZE = 28

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


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """Carga el modelo .keras previamente entrenado."""
    from tensorflow import keras

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {path}. Entrénalo con: python src/train.py"
        )
    return keras.models.load_model(path)


def preprocess_image(img: Image.Image, auto_invert: bool = True) -> np.ndarray:
    """Convierte una imagen PIL arbitraria en un tensor (1, 28, 28, 1) en [0, 1]."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # Componer sobre fondo blanco para manejar transparencias
        img = img.convert("RGBA")
        fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fondo, img)
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    # Si el borde es claro, asumimos fondo claro -> invertimos
    if auto_invert:
        borde = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        if borde.mean() > 127:
            arr = 255.0 - arr

    # Recortar al contenido (píxeles significativos) y dejar margen cuadrado
    mask = arr > 30
    if mask.any() and arr.shape[0] > IMG_SIZE:
        ys, xs = np.where(mask)
        arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        h, w = arr.shape
        lado = int(max(h, w) * 1.15)
        lienzo = np.zeros((lado, lado), dtype=np.float32)
        y0, x0 = (lado - h) // 2, (lado - w) // 2
        lienzo[y0 : y0 + h, x0 : x0 + w] = arr
        arr = lienzo

    small = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.LANCZOS
    )
    x = np.asarray(small, dtype=np.float32) / 255.0
    return x.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict(model, img: Image.Image, top_k: int = 3, auto_invert: bool = True) -> dict:
    """Devuelve la clase predicha, su confianza, el top-k y el vector de probabilidades."""
    x = preprocess_image(img, auto_invert=auto_invert)
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top = [(CLASS_NAMES[i], float(probs[i])) for i in order[:top_k]]
    return {
        "clase": top[0][0],
        "indice": int(order[0]),
        "confianza": top[0][1],
        "top_k": top,
        "probabilidades": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "entrada_28x28": x[0, :, :, 0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Predice la clase de una imagen de ropa")
    parser.add_argument("imagen", type=Path, help="Ruta de la imagen (png/jpg)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--no-invert", action="store_true", help="Desactiva la inversión automática")
    args = parser.parse_args()

    if not args.imagen.exists():
        print(f"Error: no existe la imagen {args.imagen}", file=sys.stderr)
        return 1

    model = load_model(args.model)
    res = predict(model, Image.open(args.imagen), top_k=args.top, auto_invert=not args.no_invert)
    print(f"Imagen: {args.imagen}")
    print(f"Clase predicha: {res['clase']}  (confianza: {res['confianza']:.2%})")
    print(f"Top-{args.top}:")
    for i, (nombre, p) in enumerate(res["top_k"], 1):
        print(f"  {i}. {nombre:<20} {p:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
