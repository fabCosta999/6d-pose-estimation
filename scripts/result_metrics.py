import os
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# =======================
# paths
# =======================
BASELINE_FILE = "/content/drive/MyDrive/machine_learning_project/baseline_results/all_adds_baseline.txt"
EXTENSION_FILE = "/content/drive/MyDrive/machine_learning_project/extension_results/all_adds_extension.txt"
MODELS_INFO = "/content/6d-pose-estimation/data/Linemod_preprocessed/models/models_info.yml"

OUT_DIR = "/content/drive/MyDrive/machine_learning_project/results"
os.makedirs(OUT_DIR, exist_ok=True)

# =======================
# utils
# =======================
def load_errors(path):
    errors = defaultdict(list)
    with open(path) as f:
        for line in f:
            obj_id, add = line.split()
            errors[int(obj_id)].append(float(add))
    return errors

def add_auc(errs, diameter, max_rel=0.1, n_steps=1000):
    max_err = max_rel * diameter
    xs = np.linspace(0, max_err, n_steps)
    ys = [(errs < x).mean() for x in xs]
    return np.trapz(ys, xs) / max_err

def plot_add_cdf(errs, label, max_x=100):
    errs = np.sort(errs)
    y = np.arange(1, len(errs) + 1) / len(errs)
    plt.plot(errs, y * 100, label=label)

# =======================
# load data
# =======================
baseline = load_errors(BASELINE_FILE)
extension = load_errors(EXTENSION_FILE)

with open(MODELS_INFO) as f:
    models_info = yaml.safe_load(f)

diameters = {int(k): v["diameter"] for k, v in models_info.items()}

ABS_THRESHOLDS = [10, 20]     # mm
REL_THRESHOLDS = [0.1, 0.2]    # diameter ratio

# =======================
# metrics per method
# =======================
def compute_metrics(errors):
    rows = []

    for obj_id, errs in errors.items():
        errs = np.array(errs)
        d = diameters[obj_id]

        row = {
            "obj_id": obj_id,
            "num_samples": len(errs),
            "add_mean_mm": errs.mean(),
            "add_median_mm": np.median(errs),
            "auc_0.1d": add_auc(errs, d) * 100,
        }

        for t in ABS_THRESHOLDS:
            row[f"acc_<_{t}mm"] = (errs < t).mean() * 100

        for r in REL_THRESHOLDS:
            row[f"acc_<_{int(r*100)}pct_d"] = (errs < r * d).mean() * 100

        rows.append(row)

    return pd.DataFrame(rows).sort_values("obj_id")

df_baseline = compute_metrics(baseline)
df_extension = compute_metrics(extension)

# =======================
# save tables
# =======================
df_baseline.to_csv(f"{OUT_DIR}/metrics_baseline.csv", index=False)
df_extension.to_csv(f"{OUT_DIR}/metrics_extension.csv", index=False)

# comparison table (delta)
df_cmp = df_extension.copy()
for col in df_baseline.columns:
    if col not in ["obj_id", "num_samples"]:
        df_cmp[col] = df_extension[col] - df_baseline[col]

df_cmp.to_csv(f"{OUT_DIR}/metrics_comparison.csv", index=False)

# =======================
# plot CDF (baseline vs extension)
# =======================
all_baseline = np.concatenate(list(baseline.values()))
all_extension = np.concatenate(list(extension.values()))

plt.figure(figsize=(7,5))
plot_add_cdf(all_baseline, "Baseline", max_x=100)
plot_add_cdf(all_extension, "Extension", max_x=100)

plt.xlabel("ADD error [mm]")
plt.ylabel("Accuracy [%]")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/add_cdf_baseline_vs_extension.png", dpi=200)
plt.close()

print("[INFO] Results saved in:", OUT_DIR)
