# Palmistry AI — Client Validation Portal

A client-facing Streamlit application for live camera/upload inference and presentation of the Palmistry AI validation evidence produced in Google Colab.

## Interactive prediction

- Capture a palm image with the browser camera or upload JPG/JPEG/PNG/WEBP.
- Build the production four-channel input: CLAHE-normalized BGR plus a Hessian ridge map.
- Run the validated ResNet50 multi-label checkpoint on CPU.
- Display confidence scores for all 16 target palm features.
- Validate checkpoint integrity with SHA-256 before model loading.

## Included evidence

- Internal validation history across 8 epochs
- Best validation accuracy: **82.81%** at epoch 6
- Dataset readiness audit: 274 four-channel tensors
- Tensor integrity test: 10/10 passed
- Batch inference test: 5/5 completed without error
- Real aligned image, fourth-channel ridge map, Grad-CAM, and diagnostic overlay

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Connect this GitHub repository at [share.streamlit.io](https://share.streamlit.io/).
2. Select `app.py` as the entry point.
3. Deploy. No secrets or Google Drive mount are required; the checkpoint is assembled from repository-safe parts during cold start.

## Methodology note

The reported accuracy is from the internal validation split. An independently annotated external hold-out dataset is recommended before production or domain-critical use.

Prepared by **Ikhsan Kamal — Data Analyst & Machine Learning Specialist**.
