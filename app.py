from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tflite" / "model.tflite"
LABEL_PATH = BASE_DIR / "tflite" / "label.txt"


st.set_page_config(
    page_title="Klasifikasi Kanji N5",
    page_icon="🈶",
    layout="centered",
)


@st.cache_data
def load_labels(label_path: Path) -> list[str]:
    if not label_path.exists():
        raise FileNotFoundError(f"File label tidak ditemukan: {label_path}")

    with label_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


@st.cache_resource
def load_interpreter(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def get_input_size(input_details: dict[str, Any]) -> tuple[int, int, int]:
    shape = input_details["shape"]
    if len(shape) != 4:
        raise ValueError(f"Input model tidak didukung. Shape: {shape}")

    _, height, width, channels = shape
    return int(height), int(width), int(channels)


def prepare_image(
    image: Image.Image,
    input_details: dict[str, Any],
    invert_colors: bool,
) -> np.ndarray:
    height, width, channels = get_input_size(input_details)

    if channels != 1:
        raise ValueError(
            f"Model mengharapkan {channels} channel, bukan grayscale 1 channel."
        )

    image = image.convert("L").resize((width, height))
    image_array = np.asarray(image, dtype=np.float32)

    if invert_colors:
        image_array = 255.0 - image_array

    image_array = np.expand_dims(image_array, axis=(0, -1))
    input_dtype = input_details["dtype"]

    if np.issubdtype(input_dtype, np.floating):
        return image_array.astype(input_dtype)

    scale, zero_point = input_details.get("quantization", (0.0, 0))
    if scale and scale > 0:
        image_array = image_array / scale + zero_point

    dtype_info = np.iinfo(input_dtype)
    image_array = np.rint(image_array)
    image_array = np.clip(image_array, dtype_info.min, dtype_info.max)
    return image_array.astype(input_dtype)


def dequantize_predictions(
    predictions: np.ndarray, output_details: dict[str, Any]
) -> np.ndarray:
    output_dtype = output_details["dtype"]
    predictions = predictions.astype(np.float32)

    if np.issubdtype(output_dtype, np.integer):
        scale, zero_point = output_details.get("quantization", (0.0, 0))
        if scale and scale > 0:
            predictions = scale * (predictions - zero_point)

    return predictions


def predict(
    interpreter: Any,
    image: Image.Image,
    invert_colors: bool,
) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    image_array = prepare_image(image, input_details, invert_colors)

    interpreter.set_tensor(input_details["index"], image_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details["index"])[0]
    return dequantize_predictions(predictions, output_details)


def build_prediction_table(
    predictions: np.ndarray,
    labels: list[str],
    top_k: int,
) -> pd.DataFrame:
    top_indices = np.argsort(predictions)[::-1][:top_k]

    rows = []
    for index in top_indices:
        label = labels[index] if index < len(labels) else f"Kelas {index}"
        confidence = float(predictions[index])
        rows.append(
            {
                "Label": label,
                "Confidence": confidence,
                "Confidence (%)": confidence * 100,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    st.title("🈶 Klasifikasi Kanji N5")
    st.write(
        "Upload gambar kanji, lalu aplikasi akan memprediksi kelasnya "
        "menggunakan model TF-Lite di folder `tflite/`."
    )

    with st.sidebar:
        st.header("Pengaturan")
        invert_colors = st.checkbox(
            "Inversi warna gambar",
            value=True,
            help="Aktif sesuai preprocessing notebook: tinta gelap diubah menjadi terang di atas latar gelap.",
        )
        top_k = st.slider("Jumlah prediksi teratas", min_value=1, max_value=10, value=5)
        st.caption(f"Model: `{MODEL_PATH.relative_to(BASE_DIR)}`")
        st.caption(f"Label: `{LABEL_PATH.relative_to(BASE_DIR)}`")

    try:
        labels = load_labels(LABEL_PATH)
        interpreter = load_interpreter(MODEL_PATH)
    except Exception as error:
        st.error("Gagal memuat model atau label.")
        st.exception(error)
        st.stop()

    uploaded_file = st.file_uploader(
        "Pilih gambar kanji",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded_file is None:
        st.info("Silakan upload gambar terlebih dahulu.")
        return

    image = Image.open(uploaded_file)
    predictions = predict(interpreter, image, invert_colors)
    prediction_table = build_prediction_table(predictions, labels, top_k)

    best_prediction = prediction_table.iloc[0]
    best_label = best_prediction["Label"]
    best_confidence = best_prediction["Confidence (%)"]

    col_image, col_result = st.columns([1, 1])

    with col_image:
        st.subheader("Gambar")
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

    with col_result:
        st.subheader("Hasil")
        st.metric("Prediksi utama", best_label, f"{best_confidence:.2f}%")

    st.subheader("Top prediksi")
    chart_data = prediction_table.set_index("Label")[["Confidence (%)"]]
    st.bar_chart(chart_data)

    st.dataframe(
        prediction_table.style.format(
            {
                "Confidence": "{:.4f}",
                "Confidence (%)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
