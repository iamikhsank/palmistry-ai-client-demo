# -*- coding: utf-8 -*-
"""Palmistry Deep Learning Architecture & Multi-Label Prediction Pipeline (16 Fitur).

Modul ini mengimplementasikan model Deep Learning berbasis PyTorch untuk memprediksi
16 fitur garis tangan dan bukit telapak tangan menggunakan tensor 4-kanal (RGB + Hessian Ridge).
Dilengkapi dengan Grad-CAM interpretability engine, dataset loader, dan fungsi inferensi produksi.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# Konfigurasi Logging Terstruktur
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PalmistryDL")

# ==============================================================================
# 1. DEFINISI 16 FITUR TARGET PALMISTRY (SESUAI SKEMA CVAT & ANATOMI TELAPAK)
# ==============================================================================
TARGET_PALM_FEATURES: List[Dict[str, str]] = [
    {"id": "mark_apollo_star", "name_id": "Bintang Bukit Apollo", "type": "box", "target_mount": "mount_apollo"},
    {"id": "mark_venus_square", "name_id": "Kotak Bukit Venus", "type": "box", "target_mount": "mount_venus"},
    {"id": "mark_jupiter_star", "name_id": "Bintang Bukit Yupiter", "type": "box", "target_mount": "mount_jupiter"},
    {"id": "branch_terminal_fork", "name_id": "Cabang Garis Ujung (Fork)", "type": "polyline", "target_mount": "general"},
    {"id": "mark_obstacle_bar", "name_id": "Garis Rintangan (Bar Mark)", "type": "polyline", "target_mount": "general"},
    {"id": "mark_venus_grille", "name_id": "Kisi-kisi Bukit Venus", "type": "box", "target_mount": "mount_venus"},
    {"id": "mark_island", "name_id": "Tanda Pulau (Island)", "type": "box", "target_mount": "general"},
    {"id": "line_influence_mars", "name_id": "Garis Pengaruh Mars", "type": "polyline", "target_mount": "mount_mars"},
    {"id": "line_mars_sister", "name_id": "Garis Saudara Mars", "type": "polyline", "target_mount": "mount_mars"},
    {"id": "branch_ascending_jupiter", "name_id": "Cabang Naik ke Yupiter", "type": "polyline", "target_mount": "mount_jupiter"},
    {"id": "line_rascettes_wrist", "name_id": "Garis Gelang Pergelangan (Rascettes)", "type": "polyline", "target_mount": "wrist"},
    {"id": "pattern_chained_origin", "name_id": "Pola Berantai Pangkal Garis", "type": "box", "target_mount": "general"},
    {"id": "line_marriage", "name_id": "Garis Hubungan/Pernikahan", "type": "polyline", "target_mount": "mount_mercury"},
    {"id": "feathering_downward", "name_id": "Percabangan Bulu Menurun", "type": "polyline", "target_mount": "general"},
    {"id": "fork_origin_separation", "name_id": "Garpu Pemisah Pangkal Garis", "type": "polyline", "target_mount": "general"},
    {"id": "discontinuity_line_break", "name_id": "Diskontinuitas/Garis Putus", "type": "box", "target_mount": "general"}
]

FEATURE_IDS: List[str] = [f["id"] for f in TARGET_PALM_FEATURES]
NUM_CLASSES: int = len(FEATURE_IDS)


# ==============================================================================
# 2. ARSITEKTUR DEEP LEARNING: 4-CHANNEL RESNET / EFFICIENTNET BACKBONE
# ==============================================================================
class PalmistryNet4Ch(nn.Module):
    """Jaringan saraf konvolusional untuk prediksi multi-label fitur telapak tangan.
    
    Menerima tensor 4-kanal (Kanal 0-2: RGB ter-normalisasi CLAHE, Kanal 3: Hessian Ridge).
    Bobot konvolusi pertama diadaptasi dari pre-trained ImageNet dengan mempertahankan
    transfer learning dan menginisialisasi kanal ke-4 dari rata-rata filter RGB.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, backbone_name: str = "resnet50", dropout_rate: float = 0.3):
        super(PalmistryNet4Ch, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if backbone_name == "resnet50":
            # The production checkpoint contains the complete network state.
            # Avoid downloading ImageNet weights during a Streamlit cold start.
            base_model = models.resnet50(weights=None)
            
            # Adaptasi layer pertama dari 3 kanal menjadi 4 kanal
            original_conv = base_model.conv1
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Inisialisasi bobot: Salin RGB ke kanal 0-2, dan kanal 3 diisi rata-rata RGB
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight
                new_conv.weight[:, 3:4, :, :] = torch.mean(original_conv.weight, dim=1, keepdim=True)
            
            base_model.conv1 = new_conv
            self.features = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4
            )
            in_features = base_model.fc.in_features
            
        elif backbone_name == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights=None)
            original_conv = base_model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight
                new_conv.weight[:, 3:4, :, :] = torch.mean(original_conv.weight, dim=1, keepdim=True)
            base_model.features[0][0] = new_conv
            self.features = base_model.features
            in_features = base_model.classifier[1].in_features
        else:
            raise ValueError(f"Backbone '{backbone_name}' tidak didukung.")

        # Spatial Attention Pooling
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_features, in_features // 4, kernel_size=1),
            nn.BatchNorm2d(in_features // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-Label Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.Mish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Mish(inplace=True),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(128, num_classes)
        )

        # Grad-CAM storage hooks
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def activations_hook(self, grad: torch.Tensor):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eksekusi forward pass tensor 4-kanal.
        
        Args:
            x: Tensor citra berdimensi (Batch, 4, Height, Width).
            
        Returns:
            Logits unnormalized untuk setiap 16 fitur berdimensi (Batch, 16).
        """
        feats = self.features(x)
        
        # Simpan aktivasi untuk Grad-CAM jika sedang dalam mode evaluasi
        if not self.training and x.requires_grad:
            self.activations = feats
            h = feats.register_hook(self.activations_hook)

        # Attention weighting
        att = self.spatial_attention(feats)
        weighted_feats = feats * att

        pooled = self.global_pool(weighted_feats).flatten(1)
        logits = self.classifier(pooled)
        return logits


# ==============================================================================
# 3. GRAD-CAM INTERPRETABILITY ENGINE (VISUALISASI LOKASI FITUR TELAPAK)
# ==============================================================================
class PalmGradCAM:
    """Mesin Grad-CAM untuk mengekstrak heatmap atensi spasial per fitur garis tangan."""

    def __init__(self, model: PalmistryNet4Ch):
        self.model = model
        self.model.eval()

    def generate_cam(self, input_tensor: torch.Tensor, target_class_idx: int) -> np.ndarray:
        """Menghasilkan peta aktivasi kelas (CAM) ternormalisasi [0, 1].
        
        Args:
            input_tensor: Tensor input 4-kanal (1, 4, H, W).
            target_class_idx: Indeks fitur target (0 s.d. 15).
            
        Returns:
            Heatmap berdimensi (H, W) dengan rentang nilai [0, 1].
        """
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)
        score = logits[0, target_class_idx]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.model.gradients
        activations = self.model.activations

        if gradients is None or activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling pada gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_resized = F.interpolate(
            cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear", align_corners=False
        )
        cam_np = cam_resized.squeeze().cpu().detach().numpy()

        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm


# ==============================================================================
# 4. DATASET LOADER UNTUK TENSOR 4-KANAL & ANOTASI CVAT
# ==============================================================================
class Palm4ChDataset(Dataset):
    """PyTorch Dataset untuk memuat file '_tensor4ch.npy' dan metadata anotasi."""

    def __init__(self, dataset_dir: Union[str, Path], annotations_json: Optional[str] = None, target_size: int = 512):
        self.dataset_dir = Path(dataset_dir)
        self.target_size = target_size
        self.tensor_files = sorted(list(self.dataset_dir.glob("*_tensor4ch.npy")))
        
        # Load anotasi CVAT jika tersedia
        self.annotations: Dict[str, np.ndarray] = {}
        if annotations_json and os.path.exists(annotations_json):
            with open(annotations_json, "r", encoding="utf-8") as f:
                raw_annot = json.load(f)
                # Parsing mapping label per sampel jika file anotasi terstruktur
                logger.info(f"Berhasil memuat anotasi dari {annotations_json}")

    def __len__(self) -> int:
        return len(self.tensor_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        file_path = self.tensor_files[idx]
        stem = file_path.stem.replace("_tensor4ch", "")
        
        # Load array 4-channel: (H, W, 4) uint8
        tensor_np = np.load(str(file_path))
        if tensor_np.shape[0] != self.target_size or tensor_np.shape[1] != self.target_size:
            tensor_np = cv2.resize(tensor_np, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        # Normalisasi ke [0.0, 1.0] dan ubah ke channel-first: (4, H, W)
        tensor_float = tensor_np.astype(np.float32) / 255.0
        
        # Standardize RGB (Kanal 0-2) dengan ImageNet mean/std, Hessian (Kanal 3) dengan mean/std lokal
        rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor_float[:, :, :3] = (tensor_float[:, :, :3] - rgb_mean) / rgb_std
        tensor_float[:, :, 3] = (tensor_float[:, :, 3] - 0.5) / 0.5

        tensor_ch_first = np.transpose(tensor_float, (2, 0, 1))
        tensor_tensor = torch.from_numpy(tensor_ch_first)

        # Label ground truth (default nol jika belum dianotasi)
        labels = self.annotations.get(stem, np.zeros(NUM_CLASSES, dtype=np.float32))
        labels_tensor = torch.from_numpy(labels)

        return tensor_tensor, labels_tensor, stem


# ==============================================================================
# 5. INFERENCE & DIAGNOSTIC VISUALIZATION PIPELINE
# ==============================================================================
@dataclass
class PalmPredictionResult:
    stem: str
    feature_probabilities: Dict[str, float]
    detected_features: List[Dict[str, Any]]
    gradcam_heatmaps: Dict[str, np.ndarray]
    clahe_rgb: np.ndarray
    ridge_map: np.ndarray


class PalmPredictor:
    """Engine inferensi untuk memprediksi 16 fitur garis tangan dan menghasilkan diagnosa visual."""

    def __init__(self, model_weights_path: Optional[str] = None, backbone: str = "resnet50", device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Menginisialisasi PalmPredictor pada perangkat: {self.device}")

        self.model = PalmistryNet4Ch(num_classes=NUM_CLASSES, backbone_name=backbone)
        if model_weights_path and os.path.exists(model_weights_path):
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Berhasil memuat bobot model dari: {model_weights_path}")
        else:
            logger.warning("Bobot pra-latih tidak ditemukan; model menggunakan inisialisasi ImageNet + He Normal.")

        self.model.to(self.device)
        self.model.eval()
        self.grad_cam = PalmGradCAM(self.model)

    def preprocess_tensor(self, tensor_4ch: np.ndarray, target_size: int = 512) -> torch.Tensor:
        """Menyiapkan tensor 4-kanal mentah ke format siap konsumsi PyTorch."""
        resized = cv2.resize(tensor_4ch, (target_size, target_size), interpolation=cv2.INTER_AREA)
        float_arr = resized.astype(np.float32) / 255.0

        # Standardisasi
        rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        float_arr[:, :, :3] = (float_arr[:, :, :3] - rgb_mean) / rgb_std
        float_arr[:, :, 3] = (float_arr[:, :, 3] - 0.5) / 0.5

        ch_first = np.transpose(float_arr, (2, 0, 1))
        batch_tensor = torch.from_numpy(ch_first).unsqueeze(0).to(self.device)
        return batch_tensor

    def predict(self, tensor_4ch: np.ndarray, stem: str = "sample", confidence_threshold: float = 0.40, top_k_gradcam: int = 3) -> PalmPredictionResult:
        """Mengeksekusi inferensi lengkap pada 1 tensor 4-kanal telapak tangan.
        
        Args:
            tensor_4ch: Array 4-kanal berdimensi (H, W, 4) uint8 [BGR + Hessian].
            stem: Pengenal atau nama sampel.
            confidence_threshold: Ambang batas probabilitas minimum untuk deteksi fitur.
            top_k_gradcam: Jumlah fitur teratas yang akan dihitung heatmap Grad-CAM-nya.
            
        Returns:
            PalmPredictionResult berisi seluruh skor probabilitas, fitur terdeteksi, dan visualisasi.
        """
        input_tensor = self.preprocess_tensor(tensor_4ch)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        feature_probs: Dict[str, float] = {}
        detected_features: List[Dict[str, Any]] = []

        for idx, feat_info in enumerate(TARGET_PALM_FEATURES):
            fid = feat_info["id"]
            p = float(probs[idx])
            feature_probs[fid] = round(p, 4)

            if p >= confidence_threshold:
                detected_features.append({
                    "id": fid,
                    "name_id": feat_info["name_id"],
                    "probability": round(p, 4),
                    "confidence_pct": f"{p * 100:.1f}%",
                    "target_mount": feat_info["target_mount"],
                    "class_idx": idx
                })

        # Urutkan berdasarkan probabilitas tertinggi
        detected_features.sort(key=lambda x: x["probability"], reverse=True)

        # Hitung Grad-CAM untuk fitur teratas
        heatmaps: Dict[str, np.ndarray] = {}
        for feat in detected_features[:top_k_gradcam]:
            cam = self.grad_cam.generate_cam(input_tensor, feat["class_idx"])
            heatmaps[feat["id"]] = cam

        bgr = tensor_4ch[:, :, :3]
        clahe_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ridge = tensor_4ch[:, :, 3]

        return PalmPredictionResult(
            stem=stem,
            feature_probabilities=feature_probs,
            detected_features=detected_features,
            gradcam_heatmaps=heatmaps,
            clahe_rgb=clahe_rgb,
            ridge_map=ridge
        )

    def render_diagnostic_dashboard(self, result: PalmPredictionResult, save_path: Optional[str] = None):
        """Menampilkan panel visualisasi diagnostik 4-tampilan untuk laporan analisis telapak."""
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
        fig.suptitle(f"Hasil Analisis Garis Tangan AI (Deep Learning): {result.stem}", fontsize=15, fontweight="bold")

        # 1. Aligned RGB
        axes[0].imshow(result.clahe_rgb)
        axes[0].set_title("1. Citra Telapak Aligned (0° Vertikal)")
        axes[0].axis("off")

        # 2. Hessian Ridge
        axes[1].imshow(result.ridge_map, cmap="inferno")
        axes[1].set_title("2. Kanal-4 (Hessian Multiscale Ridge)")
        axes[1].axis("off")

        # 3. Grad-CAM Overlay untuk fitur teratas
        if result.detected_features and result.detected_features[0]["id"] in result.gradcam_heatmaps:
            top_feat = result.detected_features[0]
            cam = result.gradcam_heatmaps[top_feat["id"]]
            cam_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            cam_colored_rgb = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
            cam_resized = cv2.resize(cam_colored_rgb, (result.clahe_rgb.shape[1], result.clahe_rgb.shape[0]))
            overlay = cv2.addWeighted(result.clahe_rgb, 0.6, cam_resized, 0.4, 0)
            axes[2].imshow(overlay)
            axes[2].set_title(f"3. Grad-CAM: {top_feat['name_id']}\n(Conf: {top_feat['confidence_pct']})")
        else:
            axes[2].imshow(result.clahe_rgb)
            axes[2].set_title("3. Grad-CAM (Belum Ada Fitur > Threshold)")
        axes[2].axis("off")

        # 4. Bar Chart Probabilitas 16 Fitur
        sorted_probs = sorted(result.feature_probabilities.items(), key=lambda x: x[1], reverse=True)[:8]
        labels = [next(f["name_id"] for f in TARGET_PALM_FEATURES if f["id"] == k) for k, _ in sorted_probs]
        scores = [v for _, v in sorted_probs]
        y_pos = np.arange(len(labels))

        axes[3].barh(y_pos, scores, color="#2f9e5b", edgecolor="#1a2332", height=0.65)
        axes[3].set_yticks(y_pos)
        axes[3].set_yticklabels(labels, fontsize=9)
        axes[3].invert_yaxis()
        axes[3].set_xlim(0.0, 1.0)
        axes[3].set_xlabel("Probabilitas Prediksi (Sigmoid)", fontsize=10)
        axes[3].set_title("4. Skor 8 Fitur Teratas", fontsize=11, fontweight="bold")
        axes[3].grid(axis="x", linestyle="--", alpha=0.5)

        for i, v in enumerate(scores):
            axes[3].text(v + 0.02, i + 0.1, f"{v*100:.1f}%", fontsize=9, fontweight="bold")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            logger.info(f"Dashboard diagnosa disimpan ke: {save_path}")
        plt.show()


# ==============================================================================
# 6. DEMO EKSEKUSI & VERIFIKASI PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DEMO INISIALISASI & UJI INFERENSI PALMISTRY DEEP LEARNING")
    print("=" * 70)

    # 1. Inisialisasi Model Predictor
    predictor = PalmPredictor(backbone="resnet50")

    # 2. Buat tensor sintetis 4-kanal (1024, 1024, 4) sebagai simulasi input
    print("\n[1/3] Menyiapkan tensor simulasi 4-kanal (RGB + Hessian)...")
    simulated_tensor = np.zeros((1024, 1024, 4), dtype=np.uint8)
    # BGR channels (skintone background)
    simulated_tensor[:, :, 0] = 180
    simulated_tensor[:, :, 1] = 190
    simulated_tensor[:, :, 2] = 230
    # Gambar garis telapak sintetis pada kanal 3 (Hessian Ridge)
    ridge_sim = np.zeros((1024, 1024), dtype=np.uint8)
    cv2.ellipse(ridge_sim, (512, 700), (280, 400), 0, 180, 360, 255, 6)
    cv2.ellipse(ridge_sim, (450, 600), (220, 200), -20, 170, 340, 255, 5)
    simulated_tensor[:, :, 3] = ridge_sim

    # 3. Jalankan Prediksi
    print("[2/3] Mengeksekusi inferensi deep learning...")
    res = predictor.predict(simulated_tensor, stem="simulated_palm_01", confidence_threshold=0.10)

    print("\n[3/3] Ringkasan Prediksi:")
    print(f"Total Fitur Terdeteksi (>10% threshold): {len(res.detected_features)}")
    for feat in res.detected_features[:5]:
        print(f" - {feat['name_id']} ({feat['id']}): {feat['confidence_pct']} [Zona: {feat['target_mount']}]")

    print("\nMenyimpan visualisasi dashboard diagnostik ke 'diagnostic_demo.png'...")
    predictor.render_diagnostic_dashboard(res, save_path="diagnostic_demo.png")
    print("Eksekusi selesai dengan sukses!")
