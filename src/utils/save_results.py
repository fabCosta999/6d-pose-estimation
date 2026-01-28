import numpy as np
import csv
from torchvision.utils import save_image
import os

def show_translation_results(errors_xyz_per_class, results_dir, best_per_class, worst_per_class):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/best", exist_ok=True)
    os.makedirs(f"{results_dir}/worst", exist_ok=True)

    print("\n" + "="*60)
    print("TRANSLATION ERRORS (mm)")
    print("="*70)

    all_l2 = []

    for cls in sorted(errors_xyz_per_class.keys()):
        ex = np.array(errors_xyz_per_class[cls]["x"])
        ey = np.array(errors_xyz_per_class[cls]["y"])
        ez = np.array(errors_xyz_per_class[cls]["z"])
        el2 = np.array(errors_xyz_per_class[cls]["l2"])

        all_l2.extend(el2.tolist())

        print(f"\nClass {cls}:")
        print(f"  Samples: {len(el2)}")
        print(f"  Mean |x|: {ex.mean():.2f} mm")
        print(f"  Mean |y|: {ey.mean():.2f} mm")
        print(f"  Mean |z|: {ez.mean():.2f} mm")
        print(f"  Mean L2:  {el2.mean():.2f} mm")
        print(f"  Median L2: {np.median(el2):.2f} mm")
        print(f"  Acc < 5mm:  {np.mean(el2 < 5)*100:.2f}%")
        print(f"  Acc < 10mm: {np.mean(el2 < 10)*100:.2f}%")


    csv_path = f"{results_dir}/translation_errors_per_class.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class", "samples",
            "mean_x_mm", "mean_y_mm", "mean_z_mm",
            "mean_l2_mm", "median_l2_mm",
            "acc_<5mm", "acc_<10mm"
        ])

        all_l2 = []

        for cls in sorted(errors_xyz_per_class.keys()):
            ex = np.array(errors_xyz_per_class[cls]["x"])
            ey = np.array(errors_xyz_per_class[cls]["y"])
            ez = np.array(errors_xyz_per_class[cls]["z"])
            el2 = np.array(errors_xyz_per_class[cls]["l2"])
            all_l2.extend(el2.tolist())

            writer.writerow([
                cls, len(el2),
                ex.mean(), ey.mean(), ez.mean(),
                el2.mean(), np.median(el2),
                np.mean(el2 < 5) * 100,
                np.mean(el2 < 10) * 100
            ])

        all_l2 = np.array(all_l2)
        writer.writerow([])
        writer.writerow([
            "OVERALL", len(all_l2),
            "", "", "",
            all_l2.mean(), np.median(all_l2),
            np.mean(all_l2 < 5) * 100,
            np.mean(all_l2 < 10) * 100
        ])


    for cls in best_per_class:
        save_image(
            best_per_class[cls]["rgb"],
            f"{results_dir}/best/class_{cls}_err_{best_per_class[cls]['error']:.2f}_rgb.png",
            normalize=True
        )

        save_image(
            best_per_class[cls]["pixel_map"],
            f"{results_dir}/best/class_{cls}_err_{best_per_class[cls]['error']:.2f}_pixel_map.png",
            normalize=True
        )

        save_image(
            worst_per_class[cls]["rgb"],
            f"{results_dir}/worst/class_{cls}_err_{worst_per_class[cls]['error']:.2f}_rgb.png",
            normalize=True
        )

        save_image(
            worst_per_class[cls]["pixel_map"],
            f"{results_dir}/worst/class_{cls}_err_{worst_per_class[cls]['error']:.2f}_pixel_map.png",
            normalize=True
        )

def show_rotation_results(errors_per_class, results_dir, best_per_class, worst_per_class):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/best", exist_ok=True)
    os.makedirs(f"{results_dir}/worst", exist_ok=True)

    print("\n" + "="*60)
    print("POSE ESTIMATION RESULTS (per class)")
    print("="*60)

    all_errors = []

    for cls, errs in errors_per_class.items():
        errs = np.array(errs)
        all_errors.extend(errs.tolist())

        mean_err = errs.mean()
        median_err = np.median(errs)
        acc_5 = np.mean(errs < 5) * 100
        acc_10 = np.mean(errs < 10) * 100
        acc_20 = np.mean(errs < 20) * 100

        print(f"\nClass {cls}:")
        print(f"  Samples: {len(errs)}")
        print(f"  Mean error:   {mean_err:.2f}°")
        print(f"  Median error: {median_err:.2f}°")
        print(f"  Acc < 5°:  {acc_5:.2f}%")
        print(f"  Acc < 10°: {acc_10:.2f}%")
        print(f"  Acc < 20°: {acc_20:.2f}%")

    all_errors = np.array(all_errors)

    print("\n" + "-"*60)
    print("OVERALL:")
    print(f"  Mean error:   {all_errors.mean():.2f}°")
    print(f"  Median error: {np.median(all_errors):.2f}°")
    print(f"  Acc < 10°: {np.mean(all_errors < 10)*100:.2f}%")
    print(f"  Acc < 20°: {np.mean(all_errors < 20)*100:.2f}%")
    print("="*60)

    csv_path = f"{results_dir}/results_per_class.csv"

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class",
            "samples",
            "mean_error",
            "median_error",
            "acc_<5",
            "acc_<10",
            "acc_<20"
        ])

        all_errors = []

        for cls in sorted(errors_per_class.keys()):
            errs = np.array(errors_per_class[cls])
            all_errors.extend(errs.tolist())

            writer.writerow([
                cls,
                len(errs),
                errs.mean(),
                np.median(errs),
                np.mean(errs < 5) * 100,
                np.mean(errs < 10) * 100,
                np.mean(errs < 20) * 100,
            ])

        all_errors = np.array(all_errors)

        writer.writerow([])
        writer.writerow([
            "OVERALL",
            len(all_errors),
            all_errors.mean(),
            np.median(all_errors),
            "",
            np.mean(all_errors < 10) * 100,
            np.mean(all_errors < 20) * 100,
        ])

    for cls in best_per_class:
        save_image(
            best_per_class[cls]["rgb"],
            f"{results_dir}/best/class_{cls}_err_{best_per_class[cls]['error']:.2f}.png",
            normalize=True
        )

        save_image(
            worst_per_class[cls]["rgb"],
            f"{results_dir}/worst/class_{cls}_err_{worst_per_class[cls]['error']:.2f}.png",
            normalize=True
        )

def show_pipeline_results(errors, results_dir, log):
    print(f"ADD-S mean: {errors.mean():.2f} mm")
    print(f"ADD-S median: {errors.median():.2f} mm")

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "eval_extension.csv")
    adds_txt_path = os.path.join(results_dir, "all_adds_extension.txt")

    with open(csv_path, "w", newline="") as f_csv, \
        open(adds_txt_path, "w") as f_txt:

        writer = csv.writer(f_csv)

        writer.writerow([
            "obj_id",
            "num_samples",
            "adds_mean_mm",
            "adds_median_mm",
            "bbox_missing",
            "false_positive",
            "bbox_invalid",
            "depth_missing",
        ])

        for obj_id, d in sorted(log.items()):
            adds = np.array(d["adds"])

            # ---- CSV summary ----
            writer.writerow([
                obj_id,
                d["total"],
                adds.mean() if len(adds) > 0 else np.nan,
                np.median(adds) if len(adds) > 0 else np.nan,
                d["bbox_missing"],
                d["false_positive"],
                d["bbox_invalid"],
                d["depth_missing"],
            ])

            # ---- TXT: all errors ----
            for e in adds:
                f_txt.write(f"{obj_id} {e:.6f}\n")