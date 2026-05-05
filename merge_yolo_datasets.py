from pathlib import Path
import shutil

SOCCER_ROOT = Path(r"F:\F\School\3\CS4391\Project\Data\Soccer_Ball")
KEYBOARD_ROOT = Path(r"F:\F\School\3\CS4391\Project\Data\keyboard")
CELLPHONE_ROOT = Path(r"F:\F\School\3\CS4391\Project\Data\CellPhone")
LAPTOP_ROOT = Path(r"F:\F\School\3\CS4391\Project\Data\laptop")


MOUSE_ROOT_1 = Path(r"F:\F\School\3\CS4391\Project\Data\Mouse")
MOUSE_ROOT_2 = Path(r"F:\F\School\3\CS4391\Project\Data\Mouse1")
MOUSE_ROOT_3 = Path(r"F:\F\School\3\CS4391\Project\Data\mouse2")
MOUSE_ROOT_4 = Path(r"F:\F\School\3\CS4391\Project\Data\mouse3")

PEN_ROOT_1 = Path(r"F:\F\School\3\CS4391\Project\Data\pen")
PEN_ROOT_2 = Path(r"F:\F\School\3\CS4391\Project\Data\Pen2")
PEN_ROOT_3 = Path(r"F:\F\School\3\CS4391\Project\Data\pen3")



DATASETS = [
    (SOCCER_ROOT, "soccer", 0, "soccer_ball"),
    (KEYBOARD_ROOT, "keyboard", 1, "keyboard"),
    (CELLPHONE_ROOT, "cellphone", 2, "cellphone"),
    (LAPTOP_ROOT, "laptop", 3, "laptop"),

    # All mouse folders use class_id = 4
    (MOUSE_ROOT_1, "mouse_a", 4, "mouse"),
    (MOUSE_ROOT_2, "mouse_b", 4, "mouse"),
    (MOUSE_ROOT_3, "mouse_c", 4, "mouse"),
    (MOUSE_ROOT_4, "mouse_d", 4, "mouse"),

    # All pen folders use class_id = 5
    (PEN_ROOT_1, "pen_a", 5, "pen"),
    (PEN_ROOT_2, "pen_b", 5, "pen"),
    (PEN_ROOT_3, "pen_c", 5, "pen"),
]

OUT_ROOT = Path(r"F:\F\School\3\CS4391\Project\Data\combined_data")

SPLITS = ["train", "valid", "test"]
IMG_EXTS = [".jpg", ".jpeg", ".png"]


def count_dataset(src_root: Path):
    """
    Count images and labels in one YOLO dataset.
    """
    counts = {}

    for split in SPLITS:
        img_dir = src_root / split / "images"
        lab_dir = src_root / split / "labels"

        if img_dir.exists():
            images = [
                p for p in img_dir.glob("*.*")
                if p.suffix.lower() in IMG_EXTS
            ]
        else:
            images = []

        if lab_dir.exists():
            labels = list(lab_dir.glob("*.txt"))
        else:
            labels = []

        counts[split] = {
            "images": len(images),
            "labels": len(labels),
        }

    return counts


def print_counts(title: str, counts: dict):
    print(f"\n========== {title} ==========")
    total_images = 0
    total_labels = 0

    for split in SPLITS:
        img_count = counts[split]["images"]
        lab_count = counts[split]["labels"]

        total_images += img_count
        total_labels += lab_count

        print(f"{split:5s} | images: {img_count:5d} | labels: {lab_count:5d}")

    print(f"total | images: {total_images:5d} | labels: {total_labels:5d}")


def copy_dataset(src_root: Path, prefix: str, new_class_id: int):
    copied_counts = {
        split: {"images": 0, "labels": 0, "missing_labels": 0}
        for split in SPLITS
    }

    for split in SPLITS:
        src_img_dir = src_root / split / "images"
        src_lab_dir = src_root / split / "labels"

        dst_img_dir = OUT_ROOT / split / "images"
        dst_lab_dir = OUT_ROOT / split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lab_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"Skip missing: {src_img_dir}")
            continue

        for img_path in src_img_dir.glob("*.*"):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            old_stem = img_path.stem
            new_stem = f"{prefix}_{old_stem}"

            dst_img = dst_img_dir / f"{new_stem}{img_path.suffix.lower()}"
            shutil.copy2(img_path, dst_img)
            copied_counts[split]["images"] += 1

            src_label = src_lab_dir / f"{old_stem}.txt"
            dst_label = dst_lab_dir / f"{new_stem}.txt"

            if not src_label.exists():
                print(f"Warning: missing label for {img_path.name}")
                dst_label.write_text("", encoding="utf-8")
                copied_counts[split]["missing_labels"] += 1
                continue

            new_lines = []

            for line in src_label.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # Replace old class id with new class id
                parts[0] = str(new_class_id)

                new_lines.append(" ".join(parts))

            dst_label.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            copied_counts[split]["labels"] += 1

    return copied_counts


def count_combined_dataset():
    return count_dataset(OUT_ROOT)


def main():
    print("========== Source Dataset Counts ==========")

    for root, prefix, class_id, class_name in DATASETS:
        counts = count_dataset(root)
        print_counts(f"{class_name} | class_id={class_id} | root={root}", counts)

    if OUT_ROOT.exists():
        print(f"\nOutput folder already exists: {OUT_ROOT}")
        print("Delete it first if you want a clean rebuild.")
    else:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n========== Copying Datasets ==========")

    for root, prefix, class_id, class_name in DATASETS:
        print(f"\nCopying {root} as class {class_id} ({class_name})")
        copied = copy_dataset(root, prefix, class_id)

        for split in SPLITS:
            print(
                f"{split:5s} | copied images: {copied[split]['images']:5d} | "
                f"copied labels: {copied[split]['labels']:5d} | "
                f"missing labels: {copied[split]['missing_labels']:5d}"
            )

    # Create data.yaml
    data_yaml = f"""path: {OUT_ROOT}
train: train/images
val: valid/images
test: test/images

names:
"""

    unique_classes = {}

    for _, _, class_id, class_name in DATASETS:
        unique_classes[class_id] = class_name

    for class_id in sorted(unique_classes.keys()):
        data_yaml += f"  {class_id}: {unique_classes[class_id]}\n"

    (OUT_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")

    combined_counts = count_combined_dataset()
    print_counts("Combined Dataset Counts", combined_counts)

    print("\nDone.")
    print(f"Combined dataset: {OUT_ROOT}")
    print(f"data.yaml: {OUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()