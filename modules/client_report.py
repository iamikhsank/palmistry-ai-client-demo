"""Honest presentation of the existing 16-label model's output.

Scores are screening signals, not localized measurements or calibrated certainty.
No client examples, identifiers, or private palm images are included.
"""
from __future__ import annotations


LINE_LABELS = {
    "kehidupan": "Garis kehidupan",
    "kepala": "Garis kepala",
    "asmara": "Garis asmara",
    "nasib": "Garis nasib/karier",
}
OBSERVATION_TEXT = {
    "jelas": "terlihat jelas",
    "samar": "terlihat samar",
    "tidak_terlihat": "belum terlihat pada foto",
}


def describe_manual_observations(side, observations):
    """Describe explicit human entries, never infer observations from model scores."""
    lines = [f"### Catatan pengamatan — tangan {side}",
             "Bagian ini diisi peninjau secara manual, bukan hasil prediksi model."]
    entered = 0
    for key, title in LINE_LABELS.items():
        observation = observations.get(key, "belum_ditinjau")
        if observation in OBSERVATION_TEXT:
            lines.append(f"- {title}: {OBSERVATION_TEXT[observation]}.")
            entered += 1
    if not entered:
        lines.append("Belum ada pengamatan garis utama yang dicatat.")
    lines.append("Panjang, cabang, posisi, dan usia tidak disimpulkan dari pilihan ini.")
    return "\n\n".join(lines)


def describe_screening(side, probabilities, names, threshold=0.5):
    """Return cautious, reviewable paragraphs from model scores alone."""
    selected = sorted(
        ((fid, float(score)) for fid, score in probabilities.items()
         if fid in names and float(score) >= threshold),
        key=lambda item: item[1], reverse=True,
    )
    lines = [f"### Tangan {side}"]
    if selected:
        candidates = ", ".join(names[fid] for fid, _ in selected[:5])
        lines.append(
            "Model menandai beberapa pola untuk ditinjau: " + candidates + ". "
            "Ini adalah kandidat berdasarkan skor model, bukan kepastian bahwa pola tersebut terlihat."
        )
    else:
        lines.append("Tidak ada kandidat yang melampaui ambang penyaringan. "
                     "Ini tidak membuktikan bahwa pola tidak ada.")
    lines.append(
        "Garis kehidupan, kepala, asmara, dan nasib: panjang, arah, ketebalan, "
        "percabangan, lokasi, serta perkiraan titik usia belum dapat ditentukan "
        "oleh model ini. Diperlukan pemeriksaan manual atau model yang divalidasi."
    )
    return "\n\n".join(lines)


def make_report(results):
    parts = ["# Laporan awal analisis telapak tangan", 
             "Hasil ini merupakan penyaringan model versi pertama dan memerlukan tinjauan manusia."]
    parts.extend(item["narrative"] for item in results.values())
    parts.extend(item["manual_narrative"] for item in results.values()
                 if item.get("manual_narrative"))
    parts.append("Interpretasi palmistry bersifat tradisional, tidak terbukti secara ilmiah, "
                 "dan bukan dasar diagnosis atau keputusan penting.")
    return "\n\n".join(parts) + "\n"
