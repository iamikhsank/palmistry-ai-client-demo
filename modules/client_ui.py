"""Client-facing two-hand workflow, with explicit model limitations."""
import io
import hashlib

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from model_runtime import PalmImageProcessor, predict_image
from modules.client_report import (
    DETAIL_CHOICES, LINE_LABELS, describe_manual_observations, describe_screening,
    make_copyable_report, make_report,
)
from modules.crop_preview import center_crop_preview
from modules.hand_validation import HandNotDetected, HandDetectorUnavailable
from modules.line_geometry import (analyze_ridge_candidates, detect_palm_landmarks,
                                   palm_mask_from_landmarks)
from modules.experimental_line_paths import trace_ridge_zones
from modules.palm_dl_pipeline import TARGET_PALM_FEATURES


def render_client_analysis(get_predictor):
    st.subheader("Analisis dua telapak tangan")
    st.write("Unggah atau ambil foto tangan kiri dan kanan dalam satu sesi. "
             "Letakkan telapak terbuka dan tegak di tengah foto, dengan cahaya merata.")
    st.info("Model versi pertama mengenali 16 kategori pola; posisi, panjang, "
            "dan usia garis utama belum diukur.")
    mode = st.radio("Sumber foto", ["Unggah gambar", "Kamera"], horizontal=True)
    photos = {}
    columns = st.columns(2)
    for side, col in zip(("kiri", "kanan"), columns):
        with col:
            st.markdown(f"#### Tangan {side}")
            if mode == "Kamera":
                photo = st.camera_input(f"Ambil foto tangan {side}", key=f"camera_{side}")
            else:
                photo = st.file_uploader(f"Pilih foto tangan {side}",
                    type=["jpg", "jpeg", "png", "webp"], key=f"upload_{side}")
            if photo:
                photos[side] = photo.getvalue()
                st.image(photos[side], caption=f"Foto tangan {side}", width="stretch")
                try:
                    crop, removed = center_crop_preview(photos[side])
                    if removed > 0.25:
                        st.warning("Sebagian bingkai akan terpotong saat diproses. "
                                   "Periksa agar seluruh telapak tetap terlihat.")
                    with st.expander("Lihat area yang dianalisis"):
                        st.image(crop, caption="Area persegi yang masuk ke model V1",
                                 width="stretch")
                        st.caption(f"Area di luar potongan: {removed:.0%} dari bingkai foto. "
                                   "Angka ini bukan ukuran bagian telapak yang hilang.")
                except (OSError, ValueError):
                    st.error("Pratinjau tidak tersedia. Periksa format foto dan coba lagi.")
    threshold = st.slider("Ambang penyaringan model", 0.2, 0.8, 0.5, 0.05,
        help="Ambang eksperimen, belum dikalibrasi sebagai probabilitas kebenaran.")
    signature = (threshold, tuple((side, hashlib.sha256(data).hexdigest())
                                  for side, data in sorted(photos.items())))
    if st.session_state.get("client_signature") != signature:
        st.session_state.pop("client_results", None)
        st.session_state.pop("client_errors", None)
        for side in ("kiri", "kanan"):
            for key in LINE_LABELS:
                st.session_state.pop(f"manual_{side}_{key}", None)
                st.session_state.pop(f"manual_note_{side}_{key}", None)
                for field in DETAIL_CHOICES:
                    st.session_state.pop(f"manual_detail_{side}_{key}_{field}", None)
        st.session_state["client_signature"] = signature
    if st.button("Analisis foto", type="primary", width="stretch", disabled=not photos):
        results = {}
        errors = []
        names = {item["id"]: item["name_id"] for item in TARGET_PALM_FEATURES}
        try:
            with st.spinner("Memuat model versi pertama..."):
                predictor = get_predictor()
            processor = PalmImageProcessor(target_size=1024)
            for side, data in photos.items():
                try:
                    with st.spinner(f"Memproses tangan {side}..."):
                        tensor, preview, ridge, quality = processor.prepare(io.BytesIO(data))
                        prediction = predict_image(predictor, tensor, threshold)
                    results[side] = {"narrative": describe_screening(side,
                        prediction.feature_probabilities, names, threshold),
                        "scores": prediction.feature_probabilities,
                        "preview": preview, "ridge": ridge, "quality": quality}
                except HandNotDetected as exc:
                    errors.append(f"Tangan {side}: {exc}")
                except HandDetectorUnavailable:
                    errors.append(f"Tangan {side}: pemeriksaan tangan sedang tidak tersedia. Coba lagi nanti.")
                except Exception as exc:
                    errors.append(f"Tangan {side}: {type(exc).__name__} — foto tidak dapat diproses.")
            st.session_state["client_results"] = results
            st.session_state["client_errors"] = errors
        except Exception as exc:
            st.session_state["client_results"] = {}
            st.session_state["client_errors"] = [f"Model belum dapat dimuat ({type(exc).__name__})."]
    if not photos:
        st.caption("Pilih setidaknya satu foto untuk memulai; kedua foto dapat diproses bersama.")
    for error in st.session_state.get("client_errors", []):
        st.error(error)
    results = st.session_state.get("client_results", {})
    if results:
        report_results = {side: dict(item) for side, item in results.items()}
        names = {f["id"]: f["name_id"] for f in TARGET_PALM_FEATURES}
        auto_records = {
            side: {"model_probabilities": item["scores"], "model_names": names,
                   "threshold": threshold}
            for side, item in results.items()
        }
        auto_text = make_copyable_report(auto_records)
        st.markdown("### Narasi hasil analisis")
        st.caption("Narasi otomatis untuk disalin; tidak memerlukan pengisian manual.")
        st.code(auto_text, language=None)
        st.download_button("Unduh narasi otomatis", auto_text,
                           file_name="narasi_telapak_tangan.txt", mime="text/plain")
        st.markdown("### Ringkasan hasil")
        st.caption("Rincian kategori pola yang ditandai model.")
        for side, item in results.items():
            st.markdown(item["narrative"])
            q = item["quality"]
            if q.get("raw_sharpness", float('inf')) < 2.0:
                st.warning(f"Foto tangan {side} mungkin kurang tajam. Coba ulang dengan kamera "
                           "stabil dan fokus pada garis telapak. Pemeriksaan ini masih perkiraan.")
            if q["sharpness"] < 80 or q["brightness"] < 55 or q["brightness"] > 220:
                st.warning(f"Foto tangan {side} mungkin membatasi detail. Coba foto ulang dengan "
                           "pencahayaan merata dan telapak yang jelas.")
            with st.expander(f"Detail teknis tangan {side}"):
                st.image(item["ridge"], caption="Peta ridge (bukan penanda garis utama)",
                         width="stretch", clamp=True)
                st.caption("Eksperimen geometri: kandidat pada peta ridge belum dikenali "
                           "sebagai garis utama dan belum divalidasi pada foto.")
                if st.button("Hitung kandidat geometri", key=f"geometry_{side}"):
                    try:
                        landmarks = detect_palm_landmarks(item["preview"])
                        if landmarks is None:
                            raise ValueError("Titik tangan tidak terdeteksi pada area analisis.")
                        palm_mask = palm_mask_from_landmarks(landmarks)
                        candidates = analyze_ridge_candidates(item["ridge"], palm_mask=palm_mask)
                        paths, zones = trace_ridge_zones(item["ridge"], landmarks, palm_mask)
                    except (ValueError, ImportError, RuntimeError):
                        candidates = None
                        st.warning("Area telapak belum dapat ditentukan. Ambil ulang foto dengan "
                                   "seluruh tangan terlihat dan cahaya merata.")
                    if candidates:
                        st.image(palm_mask.astype('uint8') * 255,
                                 caption="Perkiraan area telapak yang dianalisis", width="stretch")
                        st.dataframe(pd.DataFrame([
                            {"Kandidat": index + 1,
                             "Panjang relatif terhadap lebar citra": round(c["length_relative_width"], 2),
                             "Batas x kiri, y atas, x kanan, y bawah": tuple(round(v, 2) for v in c["bbox_relative"]),
                             "Kelompok persilangan kandidat": c["junction_clusters"]}
                            for index, c in enumerate(candidates)
                        ]), hide_index=True, width="stretch")
                        st.caption("Koordinat relatif terhadap citra persegi; panjang adalah total "
                                   "jalur kandidat, bukan panjang fisik atau garis utama. "
                                   "Persilangan, tekstur dan noise dapat menyerupai cabang.")
                        st.markdown("**Pelacakan jalur menurut zona referensi — eksperimen**")
                        st.caption("Penamaan zona merupakan dugaan berbasis posisi, belum validasi "
                                   "garis kehidupan, kepala, atau asmara. Persilangan dapat mengubah jalur.")
                        if zones:
                            overlay = Image.fromarray(item['preview']).convert('RGB').resize((512,512))
                            draw = ImageDraw.Draw(overlay)
                            colors = {'kehidupan':'#ff4242','kepala':'#20bfff','asmara':'#ffcb33'}
                            for label, zone in zones.items():
                                xy = paths[zone['path_index']]['xy'] * (511/255)
                                draw.line([tuple(map(float,p)) for p in xy], fill=colors[label], width=3)
                            st.image(overlay, caption="Jalur kandidat: merah kehidupan, biru kepala, kuning asmara",
                                     width="stretch")
                            st.dataframe(pd.DataFrame([
                                {'Zona referensi': label.capitalize(),
                                 'Panjang jalur / lebar telapak': round(zone['length_palm_widths'], 2),
                                 'Piksel persilangan kandidat': zone['junction_pixels']}
                                for label, zone in zones.items()]), hide_index=True, width="stretch")
                            st.caption("Ukuran mengikuti jalur kandidat dalam foto, bukan sentimeter. "
                                       "Piksel persilangan bukan jumlah cabang yang terverifikasi.")
                        else:
                            st.info("Belum ada jalur zona yang cukup terpisah untuk ditampilkan.")
                    elif candidates is not None:
                        st.info("Tidak ada kandidat geometri yang memenuhi pengaturan awal.")
                names = {f["id"]: f["name_id"] for f in TARGET_PALM_FEATURES}
                rows = [{"Pola": names.get(k, k), "Skor model": v}
                        for k, v in sorted(item["scores"].items(), key=lambda x: -x[1])]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("### Catatan pengamatan manual")
        st.caption("Opsional. Peninjau mengisi berdasarkan foto; pilihan ini bukan "
                   "prediksi AI dan tidak mengubah model.")
        options = {
            "belum_ditinjau": "Belum ditinjau",
            "jelas": "Terlihat jelas",
            "samar": "Terlihat samar",
            "tidak_terlihat": "Belum terlihat pada foto",
        }
        copy_records = {}
        names = {f["id"]: f["name_id"] for f in TARGET_PALM_FEATURES}
        for side in results:
            with st.expander(f"Catat garis utama tangan {side}"):
                observations = {}
                notes = {}
                details = {}
                for key, label in LINE_LABELS.items():
                    observations[key] = st.selectbox(
                        label, list(options), format_func=options.get,
                        key=f"manual_{side}_{key}",
                    )
                    details[key] = {}
                    for field, (field_label, choices) in DETAIL_CHOICES.items():
                        details[key][field] = st.selectbox(
                            f"{field_label} — {label.lower()}", list(choices),
                            format_func=choices.get,
                            key=f"manual_detail_{side}_{key}_{field}",
                        )
                    notes[key] = st.text_area(
                        f"Detail {label.lower()} (opsional)",
                        placeholder="Contoh: panjang, arah, atau percabangan yang diperiksa peninjau",
                        key=f"manual_note_{side}_{key}", max_chars=800,
                    )
                report_results[side]["manual_narrative"] = describe_manual_observations(
                    side, observations, notes
                )
                st.markdown(report_results[side]["manual_narrative"])
                selected = sorted(
                    ((fid, float(score)) for fid, score in results[side]["scores"].items()
                     if fid in names and float(score) >= threshold),
                    key=lambda item: item[1], reverse=True,
                )
                copy_records[side] = {
                    "observations": observations, "details": details, "notes": notes,
                    "model_probabilities": results[side]["scores"], "model_names": names,
                    "threshold": threshold,
                }
        copy_text = make_copyable_report(copy_records)
        if copy_text != auto_text:
            st.markdown("### Narasi dengan catatan tambahan")
            st.code(copy_text, language=None)
            st.download_button("Unduh narasi dengan catatan", copy_text,
                               file_name="keterangan_garis.txt", mime="text/plain")
        st.download_button("Unduh ringkasan", make_report(report_results) + "\n" + copy_text,
                           file_name="ringkasan_telapak_tangan.md", mime="text/markdown")
        st.caption("Palmistry adalah interpretasi tradisional, bukan diagnosis atau dasar keputusan penting.")
