import argparse
import os
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict




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

def plot_add_cdf(errs, label):
    errs = np.sort(errs)
    y = np.arange(1, len(errs) + 1) / len(errs)
    plt.plot(errs, y * 100, label=label)


# =======================
# metrics
# =======================
def compute_global_metrics(errors, diameters, abs_thresholds, rel_thresholds):
    all_errs = []
    rows = {}

    for obj_id, errs in errors.items():
        all_errs.extend(errs)

    all_errs = np.array(all_errs)

    # diameter weighted by number of samples
    all_ds = []
    for obj_id, errs in errors.items():
        all_ds.extend([diameters[obj_id]] * len(errs))
    all_ds = np.array(all_ds)

    rows["obj_id"] = "ALL"
    rows["num_samples"] = len(all_errs)
    rows["adds_mean_mm"] = all_errs.mean()
    rows["adds_median_mm"] = np.median(all_errs)

    # Global AUC@0.1d computed using per-sample object diameters
    max_errs = 0.1 * all_ds
    xs = np.linspace(0, max_errs.max(), 1000)
    ys = [(all_errs < x).mean() for x in xs]
    rows["auc_0.1d"] = np.trapz(ys, xs) / max_errs.max() * 100

    for t in abs_thresholds:
        rows[f"acc_<_{t}mm"] = (all_errs < t).mean() * 100

    for r in rel_thresholds:
        rows[f"acc_<_{int(r*100)}pct_d"] = (all_errs < r * all_ds).mean() * 100

    return rows



def compute_metrics(errors, diameters, abs_thresholds, rel_thresholds):
    rows = []

    for obj_id, errs in errors.items():
        errs = np.array(errs)
        d = diameters[obj_id]

        row = {
            "obj_id": obj_id,
            "num_samples": len(errs),
            "adds_mean_mm": errs.mean(),
            "adds_median_mm": np.median(errs),
            "auc_0.1d": add_auc(errs, d) * 100,
        }

        for t in abs_thresholds:
            row[f"acc_<_{t}mm"] = (errs < t).mean() * 100

        for r in rel_thresholds:
            row[f"acc_<_{int(r*100)}pct_d"] = (errs < r * d).mean() * 100

        rows.append(row)

    return pd.DataFrame(rows).sort_values("obj_id")



def main(args):
    os.makedirs(args.out_dir, exist_ok=True)


    # =======================
    # load data
    # =======================
    baseline = load_errors(args.baseline_path)
    extension = load_errors(args.extension_path)

    with open(args.models_path) as f:
        models_info = yaml.safe_load(f)

    diameters = {int(k): v["diameter"] for k, v in models_info.items()}

    df_baseline = compute_metrics(baseline, diameters, args.abs_thresholds, args.rel_thresholds)
    df_extension = compute_metrics(extension, diameters, args.abs_thresholds, args.rel_thresholds)

    df_baseline = pd.concat([
        df_baseline,
        pd.DataFrame([compute_global_metrics(baseline, diameters, args.abs_thresholds, args.rel_thresholds)])
    ])

    df_extension = pd.concat([
        df_extension,
        pd.DataFrame([compute_global_metrics(extension, diameters, args.abs_thresholds, args.rel_thresholds)])
    ])

    # =======================
    # save tables
    # =======================
    df_baseline.to_csv(f"{args.out_dir}/metrics_baseline.csv", index=False)
    df_extension.to_csv(f"{args.out_dir}/metrics_extension.csv", index=False)


    # comparison table (delta)
    df_cmp = df_extension.copy()
    for col in df_baseline.columns:
        if col not in ["obj_id", "num_samples"]:
            df_cmp[col] = df_extension[col] - df_baseline[col]

    df_cmp.to_csv(f"{args.out_dir}/metrics_comparison.csv", index=False)

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

    plt.savefig(f"{args.out_dir}/add_cdf_baseline_vs_extension.png", dpi=200)
    plt.close()

    print("[INFO] Results saved in:", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute ADD-based evaluation metrics and comparison plots "
            "for baseline and extension methods on the LineMOD dataset."
        )
    )

    parser.add_argument(
        "--baseline_path",
        type=str,
        required=True,
        help="Path to the text file containing ADD errors for the baseline method "
             "(format: <object_id> <add_error_mm> per line)."
    )

    parser.add_argument(
        "--extension_path",
        type=str,
        required=True,
        help="Path to the text file containing ADD errors for the proposed extension "
             "(format: <object_id> <add_error_mm> per line)."
    )

    parser.add_argument(
        "--models_path",
        type=str,
        default="data/Linemod_preprocessed/models/models_info.yml",
        help="Path to the models_info.yml file containing object diameters "
             "(default: data/Linemod_preprocessed/models/models_info.yml)."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="result_metrics",
        help="Output directory where CSV tables and plots will be saved."
    )

    parser.add_argument(
        "--abs_thresholds",
        type=float,
        nargs="+",
        default=[10.0, 20.0],
        help="Absolute ADD thresholds in millimeters used to compute accuracy "
             "(e.g. --abs_thresholds 10 20)."
    )

    parser.add_argument(
        "--rel_thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2],
        help="Relative ADD thresholds expressed as a fraction of the object diameter "
             "(e.g. --rel_thresholds 0.1 0.2)."
    )

    args = parser.parse_args()
    main(args)
