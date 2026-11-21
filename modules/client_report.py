"""Honest presentation of the existing 16-label model's output.

Scores are screening signals, not localized measurements or calibrated certainty.
No client examples, identifiers, or private palm images are included.
"""
from __future__ import annotations

import math


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

DETAIL_CHOICES = {
    "length": ("Panjang", {"unknown": "Belum dicatat", "long": "Panjang", "short": "Pendek"}),
    "direction": ("Arah", {"unknown": "Belum dicatat", "up": "Cenderung naik", "down": "Cenderung turun", "curved": "Melengkung", "straight": "Cenderung lurus"}),
    "continuity": ("Kondisi garis", {"unknown": "Belum dicatat", "continuous": "Bersambung", "broken": "Terputus"}),
    "branches": ("Percabangan", {"unknown": "Belum dicatat", "present": "Ada percabangan", "none": "Tidak tampak percabangan"}),
}


def make_copyable_report(records):
    """Describe model labels without inventing locations or line measurements."""
    paragraphs = ["KETERANGAN TELAPAK TANGAN"]
    for side in ("kiri", "kanan"):
        if side not in records:
            continue
        data = records[side]
        paragraphs.append(f"TANGAN {side.upper()}")
        names = data.get("model_names", {})
        threshold = data.get("threshold", 0.5)
        selected = []
        for fid, raw_score in data.get("model_probabilities", {}).items():
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if fid in names and math.isfinite(score) and score >= threshold:
                selected.append((fid, score))
        selected.sort(key=lambda item: item[1], reverse=True)
        if selected:
            labels = [names[fid] for fid, _ in selected]
            paragraphs.append(
                f"Pada foto tangan {side}, model menandai {len(labels)} kategori pola: "
                + ", ".join(labels) + ". Berikut keterangan setiap kategori berdasarkan referensi pola telapak."
            )
            for fid, _ in selected:
                description = FEATURE_EXPLANATIONS.get(fid)
                if description:
                    detail = FEATURE_DETAILS.get(fid, "").split(". ")[0].rstrip(".")
                    text = f"{names[fid]}: dalam referensi, kategori ini mengacu pada {description}."
                    if detail:
                        text += f" {detail}."
                    paragraphs.append(text)
                else:
                    paragraphs.append(f"{names[fid]}: kategori pola ini ditandai oleh model.")
        elif data.get("model_candidates"):
            paragraphs.append("Kategori pola yang ditandai model: " + ", ".join(data["model_candidates"]) + ".")
        else:
            paragraphs.append("Tidak ada kategori pola yang mencapai ambang model pada foto ini.")
        manual = []
        for key, title in LINE_LABELS.items():
            values = []
            visibility = data.get("observations", {}).get(key)
            if visibility in OBSERVATION_TEXT:
                values.append(OBSERVATION_TEXT[visibility])
            for field, (_, choices) in DETAIL_CHOICES.items():
                value = data.get("details", {}).get(key, {}).get(field, "unknown")
                if value != "unknown" and value in choices:
                    values.append(choices[value].lower())
            note = data.get("notes", {}).get(key, "").strip()
            if values or note:
                sentence = f"{title}: " + (", ".join(values) + "." if values else "")
                if note:
                    sentence += (" " if values else "") + note
                manual.append(sentence)
        if manual:
            paragraphs.append("CATATAN TAMBAHAN PENINJAU")
            paragraphs.extend(manual)
    paragraphs.append("Catatan: kategori merupakan hasil model, bukan pemetaan garis utama. "
                      "Posisi, panjang, dan usia garis belum diukur.")
    return "\n\n".join(paragraphs) + "\n"

FEATURE_EXPLANATIONS = {
    "mark_apollo_star": "tanda berbentuk bintang yang dalam referensi berkaitan dengan area Apollo",
    "mark_venus_square": "pola menyerupai kotak yang dalam referensi berkaitan dengan area Venus",
    "mark_jupiter_star": "tanda berbentuk bintang yang dalam referensi berkaitan dengan area Yupiter",
    "branch_terminal_fork": "garis yang ujungnya bercabang menyerupai garpu",
    "mark_obstacle_bar": "garis pendek yang berpotensi memotong atau melintang pada garis lain",
    "mark_venus_grille": "kumpulan garis yang menyerupai kisi-kisi pada referensi area Venus",
    "mark_island": "bagian garis yang membentuk pola menyerupai pulau atau oval",
    "line_influence_mars": "pola yang dalam referensi disebut garis pengaruh Mars",
    "line_mars_sister": "pola yang dalam referensi disebut garis saudara Mars",
    "branch_ascending_jupiter": "percabangan yang dalam referensi mengarah ke area Yupiter",
    "line_rascettes_wrist": "pola garis gelang pada area pergelangan menurut referensi",
    "pattern_chained_origin": "rangkaian bentuk menyerupai rantai pada pangkal garis",
    "line_marriage": "pola yang dalam referensi disebut garis hubungan atau pernikahan",
    "feathering_downward": "percabangan halus yang pada referensi digambarkan mengarah ke bawah",
    "fork_origin_separation": "percabangan menyerupai garpu pada pangkal garis",
    "discontinuity_line_break": "pola terputus atau diskontinuitas pada garis",
}

# Keterangan pola sebagai panduan membaca kategori, bukan pengukuran foto.
FEATURE_DETAILS = {
    "mark_apollo_star": "Bentuk bintang dalam panduan digambarkan melalui beberapa garis yang saling berpotongan. Apollo merupakan nama area pada peta telapak di bawah jari manis.",
    "mark_venus_square": "Bentuk kotak dalam panduan tersusun dari garis yang membatasi sebuah ruang. Venus adalah sebutan untuk area di sekitar pangkal ibu jari.",
    "mark_jupiter_star": "Bentuk bintang dalam panduan terdiri atas beberapa garis yang bertemu atau berpotongan. Yupiter merupakan nama area pada peta telapak di bawah jari telunjuk.",
    "branch_terminal_fork": "Dalam panduan, bentuk garpu menggambarkan satu garis yang terbagi menjadi dua atau lebih bagian pada ujungnya. Bentuk, arah, dan panjang cabang perlu dibaca dari foto secara terpisah.",
    "mark_obstacle_bar": "Dalam panduan, bar mark digambarkan sebagai garis yang melintang terhadap garis lain. Hubungan antara garis tersebut dan garis utama perlu ditentukan melalui peninjauan foto.",
    "mark_venus_grille": "Pola kisi-kisi dalam panduan terbentuk dari beberapa garis yang saling bersilang. Venus adalah sebutan untuk area di sekitar pangkal ibu jari.",
    "mark_island": "Pola pulau pada panduan berbentuk ruang tertutup menyerupai oval yang mengikuti bagian garis. Kategori ini tidak menunjukkan panjang atau posisi pulau pada foto.",
    "line_influence_mars": "Garis pengaruh Mars adalah nama kategori yang digunakan dalam referensi pembacaan telapak. Bentuk dan hubungannya dengan garis lain bergantung pada hasil pengamatan foto.",
    "line_mars_sister": "Dalam panduan, garis saudara Mars dijelaskan sebagai garis pendamping yang berkaitan dengan area garis kehidupan. Panjang dan hubungan kedua garis tidak diukur oleh kategori ini.",
    "branch_ascending_jupiter": "Dalam panduan, cabang naik digambarkan bergerak menuju area Yupiter di bawah telunjuk. Titik awal dan arah cabang perlu ditentukan secara terpisah.",
    "line_rascettes_wrist": "Rascettes merupakan sebutan dalam referensi untuk garis pada pergelangan tangan. Jumlah dan bentuk masing-masing garis tidak ditentukan oleh kategori ini.",
    "pattern_chained_origin": "Pola berantai dalam panduan menggambarkan rangkaian bentuk kecil yang saling terhubung pada suatu bagian garis. Kategori ini tidak menentukan panjang rangkaian.",
    "line_marriage": "Garis hubungan atau pernikahan adalah nama kategori dalam referensi tradisional. Hasil kategori ini tidak menentukan jumlah hubungan atau kejadian dalam kehidupan seseorang.",
    "feathering_downward": "Dalam panduan, pola ini digambarkan sebagai cabang-cabang halus dengan arah menurun dari garis. Jumlah dan posisi cabang tidak dihitung oleh kategori ini.",
    "fork_origin_separation": "Dalam panduan, pola garpu pangkal menggambarkan percabangan pada awal sebuah garis. Kategori ini tidak menentukan jarak pemisahan atau garis utama yang terlibat.",
    "discontinuity_line_break": "Diskontinuitas dalam panduan merujuk pada bagian garis yang tampak terpisah atau terputus. Kategori ini tidak menentukan posisi atau durasi suatu peristiwa.",
}


def describe_manual_observations(side, observations, notes=None):
    """Describe explicit human entries, never infer observations from model scores."""
    lines = [f"### Catatan pengamatan — tangan {side}",
             "Bagian ini diisi peninjau secara manual, bukan hasil prediksi model."]
    notes = notes or {}
    entered = 0
    for key, title in LINE_LABELS.items():
        observation = observations.get(key, "belum_ditinjau")
        if observation in OBSERVATION_TEXT:
            lines.append(f"**{title}.** Berdasarkan catatan peninjau, garis ini {OBSERVATION_TEXT[observation]}.")
            entered += 1
        note = notes.get(key, "").strip()
        if note:
            lines.append(f"Catatan tambahan peninjau untuk {title.lower()}: {note}")
            entered += 1
    if not entered:
        lines.append("Belum ada pengamatan garis utama yang dicatat.")
    lines.append("Panjang, cabang, posisi, dan usia tidak disimpulkan otomatis dari pilihan ini.")
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
            "Hasil analisis model menandai kategori pola berikut: " + candidates + "."
        )
        lines.append("**Uraian hasil analisis**")
        for fid, _ in selected:
            description = FEATURE_EXPLANATIONS.get(fid)
            if description:
                lines.append(
                    f"**{names[fid]}.** Hasil model menandai kategori ini. "
                    f"Dalam panduan visual, kategori tersebut mengacu pada {description}. "
                    f"{FEATURE_DETAILS.get(fid, '')}"
                )
            else:
                lines.append(f"**{names[fid]}.** Hasil model menandai kategori pola ini.")
    else:
        lines.append("Tidak ada kandidat yang melampaui ambang penyaringan. "
                     "Ini tidak membuktikan bahwa pola tidak ada.")
    lines.append(
        "Catatan hasil: kategori di atas merupakan keluaran model, bukan kepastian bentuk "
        "atau lokasi pada foto. Panjang, arah, ketebalan, percabangan dan titik usia "
        "pada garis utama belum dapat ditentukan otomatis; catatan peninjau ditampilkan terpisah."
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
