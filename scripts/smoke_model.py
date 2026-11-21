"""Verify packaged checkpoint and CPU inference without private client data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from model_runtime import assemble_checkpoint, load_production_predictor, predict_image

checkpoint = assemble_checkpoint()
print(f"Checkpoint integrity OK: {checkpoint.name}")
predictor = load_production_predictor()
result = predict_image(predictor, np.zeros((512, 512, 4), dtype=np.uint8), 0.5)
assert len(result.feature_probabilities) == 16
assert all(np.isfinite(x) and 0 <= x <= 1 for x in result.feature_probabilities.values())
print("CPU inference completed; synthetic input, not an accuracy test.")
