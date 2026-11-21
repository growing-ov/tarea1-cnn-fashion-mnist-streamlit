"""
Interfaz web del clasificador de prendas (Fashion-MNIST) construido con una CNN.

    streamlit run app.py

La app no entrena nada: se limita a cargar el modelo que ya está en `models/` y a responder
preguntas del usuario sobre una imagen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.predict import CLASS_NAMES, DEFAULT_MODEL_PATH, load_model, predict  # noqa: E402

st.set_page_config(page_title="Clasificador CNN · Fashion-MNIST", page_icon="👕", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo entrenado…")
def get_model():
    """Devuelve el modelo entrenado, cargándolo del disco una sola vez por sesión."""
    return load_model(DEFAULT_MODEL_PATH)


st.title("👕 Clasificador de prendas con una CNN")
st.write(
    "Sube la foto de una prenda o de un calzado y la red convolcional dirá de qué tipo se trata. "
    "El modelo se entrenó sobre **Fashion-MNIST**, así que reconoce estas diez categorías: "
    + ", ".join(f"*{c}*" for c in CLASS_NAMES)
    + "."
)

with st.sidebar:
    st.header("Opciones")
    auto_invert = st.checkbox(
        "Invertir automáticamente fondos claros",
        value=True,
        help="Fashion-MNIST tiene la prenda clara sobre fondo negro. "
        "Si tu foto tiene fondo claro, se invierte para parecerse al dataset.",
    )
    st.caption(f"Modelo: `{DEFAULT_MODEL_PATH.relative_to(ROOT)}`")
    samples_dir = ROOT / "samples"
    samples = sorted(samples_dir.glob("*.png")) if samples_dir.exists() else []
    sample_choice = st.selectbox(
        "…o usa una imagen de ejemplo",
        ["(ninguna)"] + [p.name for p in samples],
    )

try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader("Cargar imagen", type=["png", "jpg", "jpeg", "bmp", "webp"])

image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif sample_choice != "(ninguna)":
    image = Image.open(samples_dir / sample_choice)

if image is None:
    st.info("⬆️ Carga una imagen (o elige un ejemplo en la barra lateral) para obtener la predicción.")
    st.stop()

res = predict(model, image, top_k=3, auto_invert=auto_invert)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.subheader("Imagen original")
    st.image(image, width="stretch")
with col2:
    st.subheader("Entrada al modelo")
    entrada = (res["entrada_28x28"] * 255).astype(np.uint8)
    st.image(Image.fromarray(entrada).resize((224, 224), Image.NEAREST), width="stretch")
    st.caption("28×28 escala de grises, normalizada a [0, 1]")
with col3:
    st.subheader("Resultado")
    st.metric("Clase predicha", res["clase"], f"confianza {res['confianza']:.1%}")
    st.progress(res["confianza"])
    st.markdown("**Top-3**")
    for i, (nombre, p) in enumerate(res["top_k"], 1):
        st.write(f"{i}. **{nombre}** — {p:.2%}")

st.subheader("Probabilidad por clase")
df = pd.DataFrame(
    {"Clase": list(res["probabilidades"].keys()), "Probabilidad": list(res["probabilidades"].values())}
).set_index("Clase")
st.bar_chart(df)

if res["confianza"] < 0.5:
    st.warning(
        "Confianza baja: la imagen puede diferir mucho de Fashion-MNIST "
        "(prenda centrada, una sola pieza, fondo uniforme)."
    )

with st.expander("¿Cómo se ha procesado la imagen?"):
    st.markdown(
        "1. Conversión a escala de grises.\n"
        "2. Inversión automática si el fondo es claro.\n"
        "3. Recorte del objeto y relleno hasta dejar un lienzo cuadrado.\n"
        "4. Redimensionado a 28×28 y normalización a [0, 1].\n"
    )
    st.caption("El mismo preprocesamiento que aplica `src/predict.py`.")
