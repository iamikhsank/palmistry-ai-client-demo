"""Evidence-linked experimental descriptions; never use for clinical claims."""
import numpy as np
from scipy.spatial import cKDTree

from modules.experimental_line_paths import trace_ridge_zones

LINE_NAMES = {'kehidupan': 'Garis kehidupan', 'kepala': 'Garis kepala', 'asmara': 'Garis asmara'}


def _similar(a, b, size, tolerance=0.035, length_tolerance=0.25):
    """Compare the actual polylines, not only the label or a bounding box."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1:] != (2,) or b.shape[1:] != (2,) or min(len(a),len(b)) < 2:
        return False
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    distance = max(float(np.percentile(da,90)), float(np.percentile(db,90))) / size
    length_a = np.linalg.norm(np.diff(a,axis=0),axis=1).sum()
    length_b = np.linalg.norm(np.diff(b,axis=0),axis=1).sum()
    return bool(distance <= tolerance and abs(length_a-length_b) / max(length_a,length_b,1) <= length_tolerance)


def stable_zones(ridge, landmarks, palm_mask, size=256, percentiles=(80,85,90)):
    """Only keep candidate paths consistent under these threshold changes.

    Stability is not a probability or an accuracy estimate. These thresholds
    are research heuristics and require independent labelled validation.
    """
    if len(percentiles) != 3 or tuple(sorted(percentiles)) != tuple(percentiles):
        raise ValueError('Three ordered thresholds required')
    runs = [trace_ridge_zones(ridge,landmarks,palm_mask,size=size,percentile=p)
            for p in percentiles]
    base_paths, base_zones = runs[1]
    stable = {}
    for name, zone in base_zones.items():
        path = base_paths[zone['path_index']]['xy']
        if all(name in zones and _similar(path, paths[zones[name]['path_index']]['xy'],size)
               for paths,zones in (runs[0],runs[2])):
            stable[name] = zone
    return base_paths, stable


def describe_geometry(side, zones):
    """Produce copyable text only from supplied geometric evidence."""
    if side not in ('kiri','kanan'):
        raise ValueError('Invalid hand side')
    lines = [f'KETERANGAN TELAPAK TANGAN {side.upper()}',
             'Keterangan berikut adalah kandidat jalur berdasarkan zona referensi; '
             'identitas garis dan kecocokannya dengan foto belum tervalidasi.']
    detected = []
    for key,title in LINE_NAMES.items():
        zone = zones.get(key)
        if zone is None:
            lines.append(f'{title}: jalur belum cukup konsisten untuk dijelaskan.')
            continue
        length = float(zone['length_palm_widths'])
        u,v = map(float,zone['position_palm'])
        if not np.isfinite([length,u,v]).all() or length <= 0:
            raise ValueError('Invalid path measurement')
        location = 'bagian atas' if v < .35 else 'bagian tengah' if v < .7 else 'bagian bawah'
        cross = ('Terdapat titik persilangan pada komponen kandidat; cabang belum dapat dipastikan.'
                 if zone['junction_pixels'] else 'Percabangan belum dapat dipastikan dari jalur ini.')
        lines.append(f'{title}: kandidat jalur berada di {location} telapak. '
                     f'Panjang jalur yang terlacak sekitar {length:.2f} kali lebar telapak pada foto. {cross}')
        detected.append(title.lower())
    if detected:
        lines.append('Kesimpulan: jalur yang konsisten pada pengaturan uji tersedia pada zona '
                     + ', '.join(detected) + '. Kestabilan ini belum membuktikan ketepatan identifikasi.')
    else:
        lines.append('Kesimpulan: belum ada jalur yang cukup konsisten untuk dirangkum. '
                     'Coba foto telapak terbuka dengan pencahayaan merata dan garis terlihat jelas.')
    return '\n\n'.join(lines)
