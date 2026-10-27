import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ----------------------------- CONFIG ---------------------------------
DATASET_ROOT = Path("C:/Users/nisha/Downloads/archive/RDD_SPLIT")            # folder containing train/ val/ test/
OUT_ROOT = Path("C:/Users/nisha/Downloads/archive/RDD_SPLIT/rdd_subset")     # where the small subset is built

N_TRAIN, N_VAL, N_TEST = 10000, 1000, 1000   # subset sizes (images); keep val/test the same as run 1
NEG_FRACTION = 0.10                          # extra images with no damage (reduces false alarms)
COUNTRIES = None                             # e.g. ["India", "Japan"] to filter by filename prefix
CLASSES = ["longitudinal_crack", "transverse_crack", "alligator_crack",
           "marking_or_manhole", "pothole"]  # order = class IDs 0-4 (checked with check_crops.py)
SEED = 42

PRIORITY_CLASS = 4                           # class ID (pothole) whose images are ALL kept first in the TRAIN subset
RUN_NAME = "rdd_subset_v2"                   # new folder, so run 1 results are not overwritten
WEIGHTS_OUT = "models/road_hazard_v2.pt"     # run 1 weights stay in models/road_hazard.pt

MODEL = "yolov8s.pt"                         # yolov8n.pt = faster, yolov8s.pt = better accuracy
EPOCHS = 50
IMGSZ = 800                                  # larger input helps small objects like potholes
BATCH = 8                                    # 6 GB GPU at 800 px; use 4 if you get an out-of-memory error
# -----------------------------------------------------------------------

IMG_EXT = {".jpg", ".jpeg", ".png"}
random.seed(SEED)


def find_dirs(split):
    """Supports  root/split/images + root/split/labels  OR  root/images/split + root/labels/split."""
    options = [
        (DATASET_ROOT / split / "images", DATASET_ROOT / split / "labels"),
        (DATASET_ROOT / "images" / split, DATASET_ROOT / "labels" / split),
    ]
    for img_dir, lbl_dir in options:
        if img_dir.exists():
            return img_dir, lbl_dir
    raise FileNotFoundError(f"Could not find images for split '{split}' under {DATASET_ROOT}")


def get_yolo_lines(img_path, lbl_dir):
    """Return YOLO-format label lines for an image (converting VOC XML if needed)."""
    txt = lbl_dir / f"{img_path.stem}.txt"
    xml = lbl_dir / f"{img_path.stem}.xml"
    if not xml.exists():
        xml = img_path.with_suffix(".xml")   # XML sometimes sits next to the image

    if txt.exists():
        lines = []
        for ln in txt.read_text().splitlines():
            parts = ln.split()
            if len(parts) == 5 and int(parts[0]) < len(CLASSES):
                lines.append(ln.strip())
        return lines

    if xml.exists():
        root = ET.parse(xml).getroot()
        w = float(root.find("size/width").text)
        h = float(root.find("size/height").text)
        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if name not in CLASSES:
                continue
            b = obj.find("bndbox")
            x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
            x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    return None   # no label file at all


def place_image(src, dst):
    """Symlink to save disk space; fall back to copying."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def build_subset(split, n, prioritize=None):
    """
    prioritize: class ID. When set, every image containing that class is kept first
    (up to n), then the rest is filled randomly. Use it for TRAIN only, so that
    val/test keep their natural class mix and the evaluation stays honest.
    With prioritize=None the selection is identical to run 1.
    """
    img_dir, lbl_dir = find_dirs(split)
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    if COUNTRIES:
        imgs = [p for p in imgs if any(p.name.startswith(c) for c in COUNTRIES)]
    random.shuffle(imgs)

    priority, positives, negatives = [], [], []
    n_neg = int(n * NEG_FRACTION)
    n_pos = n - n_neg

    for p in imgs:
        if prioritize is None and len(positives) >= n_pos and len(negatives) >= n_neg:
            break
        lines = get_yolo_lines(p, lbl_dir)
        if lines is None:
            continue
        if lines:
            if prioritize is not None and any(int(l.split()[0]) == prioritize for l in lines):
                priority.append((p, lines))
            elif prioritize is not None or len(positives) < n_pos:
                positives.append((p, lines))
        elif len(negatives) < n_neg:
            negatives.append((p, lines))

    if prioritize is not None:
        positives = (priority + positives)[:n_pos]
        print(f"[{split}] priority class {prioritize}: {min(len(priority), n_pos)} images kept first")

    chosen = positives + negatives
    for p, lines in chosen:
        place_image(p, OUT_ROOT / "images" / split / p.name)
        lbl_out = OUT_ROOT / "labels" / split / f"{p.stem}.txt"
        lbl_out.parent.mkdir(parents=True, exist_ok=True)
        lbl_out.write_text("\n".join(lines))

    print(f"[{split}] {len(positives)} damaged + {len(negatives)} background images")


def write_yaml():
    cfg = {
        "path": str(OUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(CLASSES)},
    }
    yaml_path = OUT_ROOT / "data.yaml"
    yaml_path.write_text(yaml.dump(cfg))
    return yaml_path


def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split, n in [("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)]:
        build_subset(split, n, prioritize=PRIORITY_CLASS if split == "train" else None)
    data_yaml = write_yaml()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not found - your PyTorch is the CPU-only build.\n"
            "Install the CUDA build (see instructions) and re-run."
        )
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,             # first GPU
        patience=10,          # stop early if val mAP stops improving
        workers=4,
        seed=SEED,
        project="runs",
        name=RUN_NAME,
        exist_ok=True,
    )

    # Ultralytics decides the real output folder (e.g. runs/detect/runs/rdd_subset),
    # so read it from the trainer instead of hard-coding it.
    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    # Final evaluation on the held-out test subset
    best = YOLO(str(best_weights))
    m = best.val(data=str(data_yaml), split="test", device=0, imgsz=IMGSZ)
    print("\n===== TEST RESULTS =====")
    print(f"Precision : {m.box.mp:.3f}")
    print(f"Recall    : {m.box.mr:.3f}")
    print(f"mAP@50    : {m.box.map50:.3f}")
    print(f"mAP@50-95 : {m.box.map:.3f}")
    for i, name in enumerate(CLASSES):
        print(f"  {name} mAP@50: {m.box.ap50[i]:.3f}")

    Path("models").mkdir(exist_ok=True)
    shutil.copy(best_weights, WEIGHTS_OUT)
    print(f"\nSaved weights to {WEIGHTS_OUT}")


if __name__ == "__main__":
    main()