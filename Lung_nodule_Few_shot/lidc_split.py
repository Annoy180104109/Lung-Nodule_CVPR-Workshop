import os
import json
import random
import argparse


def split_dataset(data_root,
                  output_root,
                  val_ratio=0.15,
                  test_ratio=0.15,
                  seed=42):

    json_path = os.path.join(data_root, "nodule_sorted.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find {json_path}")

    # Create output directory if it does not exist
    os.makedirs(output_root, exist_ok=True)

    with open(json_path, "r") as f:
        meta = json.load(f)

    # ---- unique patients ----
    patients = sorted({m["patient_id"] for m in meta})
    random.Random(seed).shuffle(patients)

    n = len(patients)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test:n_test + n_val])
    train_patients = set(patients[n_test + n_val:])

    train_data, val_data, test_data = [], [], []

    for m in meta:
        pid = m["patient_id"]

        if pid in train_patients:
            train_data.append(m)
        elif pid in val_patients:
            val_data.append(m)
        else:
            test_data.append(m)

    # ---- save splits ----
    with open(os.path.join(output_root, "train.json"), "w") as f:
        json.dump(train_data, f, indent=4)

    with open(os.path.join(output_root, "val.json"), "w") as f:
        json.dump(val_data, f, indent=4)

    with open(os.path.join(output_root, "test.json"), "w") as f:
        json.dump(test_data, f, indent=4)

    print("Split complete.")
    print(f"Train nodules: {len(train_data)}")
    print(f"Val nodules:   {len(val_data)}")
    print(f"Test nodules:  {len(test_data)}")
    print(f"Saved to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to LIDC_full_annotation folder")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Where to save split json files")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    split_dataset(
        data_root=args.data_root,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )