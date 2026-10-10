from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_runtime import PalmImageProcessor, load_production_predictor, predict_image
from modules.palm_dl_pipeline import TARGET_PALM_FEATURES


BASE_DIR = Path(__file__).parent
ASSET_DIR = BASE_DIR / "assets"

st.set_page_config(
    page_title="Palmistry AI | Client Validation",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f8fc; color: #14213d; }
      [data-testid="stSidebar"] { background: #0b132b; }
      [data-testid="stSidebar"] * { color: #f8fafc; }
      .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
      .hero {
        padding: 2.2rem 2.4rem; border-radius: 24px;
        background: linear-gradient(135deg,#101a3a 0%,#163b7a 52%,#0d9488 100%);
        color: white; box-shadow: 0 20px 50px rgba(18,55,110,.18); margin-bottom: 1.25rem;
      }
      .hero h1 { color: white; margin: 0 0 .4rem 0; font-size: 2.4rem; }
      .hero p { margin: 0; opacity: .9; font-size: 1.05rem; }
      .eyebrow { letter-spacing: .12em; font-size: .78rem; font-weight: 700; opacity: .75; }
      .metric-card {
        background: white; padding: 1.1rem 1.2rem; border: 1px solid #e6ebf4;
        border-radius: 16px; min-height: 116px; box-shadow: 0 8px 24px rgba(15,23,42,.05);
      }
      .metric-label { color: #64748b; font-size: .86rem; font-weight: 650; }
      .metric-value { color: #0f2452; font-size: 1.82rem; line-height: 1.2; font-weight: 800; margin-top:.3rem; }
      .metric-note { color: #0f766e; font-size: .79rem; margin-top:.25rem; }
      .section-note { padding: .85rem 1rem; border-left: 4px solid #2563eb; background:#eff6ff; border-radius:8px; }
      .status-ok { color:#047857; font-weight:700; }
      div[data-testid="stDataFrame"] { background:white; border-radius:14px; overflow:hidden; }
      h2, h3 { color:#102a56; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.DataFrame(json.loads((ASSET_DIR / "training_metrics.json").read_text()))


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_predictor():
    return load_production_predictor()


metrics = load_metrics()
best = metrics.loc[metrics["val_accuracy"].idxmax()]

with st.sidebar:
    st.markdown("## ✋ Palmistry AI")
    st.caption("Client Validation Portal")
    st.divider()
    page = st.radio(
        "Navigasi",
        ["Live Prediction", "Executive Summary", "Model Performance", "Inference Evidence", "Methodology"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Model**  ")
    st.caption("ResNet50 · 4-channel · Multi-label")
    st.markdown("**Prepared by**  ")
    st.caption("Ikhsan Kamal  ")
    st.caption("Data Analyst & Machine Learning Specialist")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">INTERACTIVE MODEL DEMO</div>
      <h1>Palmistry AI</h1>
      <p>Camera and image-upload inference powered by the validated 4-channel ResNet50 checkpoint.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Live Prediction":
    st.subheader("Coba model dengan foto telapak tangan")
    st.markdown(
        "Ambil foto langsung dari kamera atau unggah gambar. Posisikan **telapak terbuka, "
        "tegak, di tengah**, dengan pencahayaan merata dan latar yang tidak ramai."
    )

    input_mode = st.radio(
        "Sumber gambar",
        ["Kamera", "Upload gambar"],
        horizontal=True,
    )
    if input_mode == "Kamera":
        image_input = st.camera_input("Ambil foto telapak tangan")
    else:
        image_input = st.file_uploader(
            "Upload JPG, JPEG, PNG, atau WEBP",
            type=["jpg", "jpeg", "png", "webp"],
        )

    threshold = st.slider(
        "Ambang confidence",
        min_value=0.20,
        max_value=0.80,
        value=0.40,
        step=0.05,
        help="Fitur dengan skor di atas nilai ini ditandai sebagai terdeteksi.",
    )

    if image_input is None:
        st.info("Aktifkan kamera atau unggah gambar untuk memulai prediksi.")
    else:
        processor = PalmImageProcessor(target_size=1024)
        try:
            with st.spinner("Menyiapkan citra 4-kanal (CLAHE + Hessian ridge)..."):
                tensor_4ch, preview_rgb, ridge_map, quality = processor.prepare(image_input)

            preview_col, ridge_col = st.columns(2)
            with preview_col:
                st.image(preview_rgb, caption="Citra terproses dan dinormalisasi", width="stretch")
            with ridge_col:
                st.image(ridge_map, caption="Hessian ridge map — kanal ke-4", width="stretch", clamp=True)

            q1, q2 = st.columns(2)
            q1.metric("Brightness", f"{quality['brightness']:.0f} / 255")
            q2.metric("Sharpness", f"{quality['sharpness']:.0f}")
            if quality["brightness"] < 55 or quality["brightness"] > 220:
                st.warning("Pencahayaan kurang ideal. Coba ambil ulang foto dengan cahaya lebih merata.")
            if quality["sharpness"] < 80:
                st.warning("Foto terlihat kurang tajam. Stabilkan kamera dan fokuskan pada garis telapak.")

            if st.button("Jalankan Prediksi AI", type="primary", width="stretch"):
                with st.spinner("Memuat checkpoint ResNet50 dan menjalankan inferensi..."):
                    predictor = get_predictor()
                    result = predict_image(predictor, tensor_4ch, threshold)

                name_by_id = {item["id"]: item["name_id"] for item in TARGET_PALM_FEATURES}
                ranked = sorted(
                    result.feature_probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                top_id, top_score = ranked[0]

                st.success("Prediksi selesai menggunakan checkpoint produksi.")
                r1, r2, r3 = st.columns(3)
                r1.metric("Prediksi terkuat", name_by_id[top_id])
                r2.metric("Confidence tertinggi", f"{top_score * 100:.1f}%")
                r3.metric("Fitur terdeteksi", str(len(result.detected_features)))

                result_table = pd.DataFrame(
                    [
                        {
                            "Fitur": name_by_id[feature_id],
                            "Confidence (%)": probability * 100,
                            "Terdeteksi": "Ya" if probability >= threshold else "Tidak",
                        }
                        for feature_id, probability in ranked
                    ]
                )
                st.markdown("### Hasil 16 fitur")
                st.dataframe(
                    result_table.style.format({"Confidence (%)": "{:.1f}%"}),
                    hide_index=True,
                    width="stretch",
                )

                chart_data = result_table.head(8).set_index("Fitur")[["Confidence (%)"]]
                st.bar_chart(chart_data, horizontal=True, color="#0d9488")
                st.caption(
                    "Hasil merupakan keluaran model multi-label untuk demonstrasi teknis, "
                    "bukan diagnosis medis atau dasar keputusan penting."
                )
        except Exception as exc:
            st.error("Gambar belum dapat diproses. Coba foto lain dengan telapak lebih jelas dan tegak.")
            with st.expander("Detail teknis"):
                st.code(f"{type(exc).__name__}: {exc}")

elif page == "Executive Summary":
    st.subheader("Validation at a glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Best validation accuracy", f"{best.val_accuracy:.2f}%", "Internal validation split")
    with c2:
        metric_card("Dataset tensors", "274", "4-channel · 1024 × 1024")
    with c3:
        metric_card("Integrity checks", "10 / 10", "100% passed")
    with c4:
        metric_card("Batch inference", "5 / 5", "Completed without error")

    st.markdown("### What was validated")
    left, right = st.columns([1.1, 1])
    with left:
        audit = pd.DataFrame(
            [
                ["4-channel tensors", 274, "READY"],
                ["Aligned 1024×1024 images", 274, "READY"],
                ["Ridge / line maps", 274, "READY"],
                ["Production checkpoint", 1, "READY"],
                ["Training history", 1, "READY"],
            ],
            columns=["Component", "Count", "Status"],
        )
        st.dataframe(audit, hide_index=True, width="stretch")
    with right:
        st.markdown(
            """
            <div class="section-note">
              <b>Outcome</b><br><br>
              The processed dataset, saved checkpoint, and training history were readable. The production
              checkpoint completed inference on five deterministic samples, and the tensor format passed all
              numerical integrity checks.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success("Technical validation completed in Google Colab")

    st.markdown("### Diagnostic overview")
    st.image(ASSET_DIR / "full_diagnostic.png", width="stretch")
    st.caption("Real pipeline output: source image, alignment, ridge channel, Grad-CAM attention, and overlay.")

elif page == "Model Performance":
    st.subheader("Training & validation performance")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Best accuracy", f"{best.val_accuracy:.2f}%", f"Epoch {int(best.epoch)}")
    with c2:
        metric_card("Best validation loss", f"{metrics.val_loss.min():.4f}", "Lowest across 8 epochs")
    with c3:
        metric_card("Final validation accuracy", f"{metrics.iloc[-1].val_accuracy:.2f}%", "Epoch 8")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics.epoch, y=metrics.train_loss, name="Train loss", mode="lines+markers", line=dict(color="#2563eb", width=3)))
    fig.add_trace(go.Scatter(x=metrics.epoch, y=metrics.val_loss, name="Validation loss", mode="lines+markers", line=dict(color="#f97316", width=3)))
    fig.update_layout(title="Loss by epoch", xaxis_title="Epoch", yaxis_title="Loss", height=390, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=metrics.epoch, y=metrics.val_accuracy, name="Validation accuracy", mode="lines+markers", line=dict(color="#059669", width=4), fill="tozeroy", fillcolor="rgba(5,150,105,.10)"))
    fig2.add_trace(go.Scatter(x=[best.epoch], y=[best.val_accuracy], name="Best checkpoint", mode="markers", marker=dict(color="#dc2626", size=13)))
    fig2.update_layout(title="Multi-label validation accuracy", xaxis_title="Epoch", yaxis_title="Accuracy (%)", height=390, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig2, width="stretch")

    st.info("Accuracy shown here is from the internal validation split. An independently annotated external hold-out set is recommended before production use.")

elif page == "Inference Evidence":
    st.subheader("Batch inference evidence")
    inference = pd.DataFrame(
        [
            ["IMG-20260531-WA0039", 9, "Percabangan Bulu Menurun", 72.18, "SUCCESS"],
            ["IMG-20260531-WA0063", 12, "Kisi-kisi Bukit Venus", 84.16, "SUCCESS"],
            ["IMG_20260909_141313_701", 10, "Tanda Pulau (Island)", 77.74, "SUCCESS"],
            ["IMG_20260918_171004_754", 13, "Tanda Pulau (Island)", 68.58, "SUCCESS"],
            ["IMG_20260918_171006_751", 15, "Tanda Pulau (Island)", 62.74, "SUCCESS"],
        ],
        columns=["Sample ID", "Detected features", "Top prediction", "Confidence (%)", "Status"],
    )
    st.dataframe(
        inference.style.format({"Confidence (%)": "{:.2f}%"}),
        hide_index=True,
        width="stretch",
    )
    st.success("5/5 samples were processed without runtime error.")

    st.markdown("### Visual evidence from the inference pipeline")
    st.image(ASSET_DIR / "full_diagnostic.png", width="stretch")

    st.markdown("### Fourth-channel construction")
    a, b = st.columns(2)
    with a:
        st.image(ASSET_DIR / "sample_aligned.jpg", caption="Aligned palm image", width="stretch")
    with b:
        st.image(ASSET_DIR / "sample_ridge.png", caption="Derived ridge map (channel 4)", width="stretch")

elif page == "Methodology":
    st.subheader("Validation methodology")
    st.markdown(
        """
        1. **Dataset readiness** — verify that tensors, aligned images, ridge maps, model checkpoint, and metric history are available.
        2. **Tensor integrity** — sample ten tensors deterministically and validate shape `(1024, 1024, 4)`, numeric finiteness, type, and range.
        3. **Checkpoint loading** — initialize the ResNet50 multi-label classifier using the saved production checkpoint.
        4. **Batch inference** — process five deterministic dataset samples with a 0.40 confidence threshold.
        5. **Explainability** — generate Grad-CAM for the strongest prediction and compare it with the aligned image and ridge channel.
        """
    )

    integrity = pd.DataFrame(
        [
            ["IMG-20260531-WA0039", "(1024, 1024, 4)", "uint8", "PASS"],
            ["IMG-20260531-WA0063", "(1024, 1024, 4)", "uint8", "PASS"],
            ["IMG_20260909_141313_701", "(1024, 1024, 4)", "uint8", "PASS"],
            ["IMG_20260918_171004_754", "(1024, 1024, 4)", "uint8", "PASS"],
            ["IMG_20260918_171006_751", "(1024, 1024, 4)", "uint8", "PASS"],
            ["ext11k_Hand_0000117", "(1024, 1024, 4)", "uint8", "PASS"],
            ["ext11k_Hand_0000171", "(1024, 1024, 4)", "uint8", "PASS"],
            ["ext11k_Hand_0000213", "(1024, 1024, 4)", "uint8", "PASS"],
            ["ext11k_Hand_0000270", "(1024, 1024, 4)", "uint8", "PASS"],
            ["ext11k_Hand_0000386", "(1024, 1024, 4)", "uint8", "PASS"],
        ],
        columns=["Sample", "Shape", "Type", "Result"],
    )
    st.markdown("### Integrity sample")
    st.dataframe(integrity, hide_index=True, width="stretch")
    st.markdown(
        """
        ### Recommended next step

        Build an expert-annotated external hold-out set, then report precision, recall, F1-score, and calibrated threshold per palm feature. This separates **technical pipeline readiness** from **clinical or domain validity**.
        """
    )

st.divider()
st.caption("Palmistry AI Client Validation · Prepared by Ikhsan Kamal · Evidence generated from the Google Colab validation workflow")
