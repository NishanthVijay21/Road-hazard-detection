import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

img_dir, lbl_dir = Path(sys.argv[1]), Path(sys.argv[2])
SAMPLES_PER_CLASS = 4
random.seed(0)

counts = defaultdict(int)
examples = defaultdict(list)      # class id -> list of (image path, boxes of that class)

label_files = list(lbl_dir.glob("*.txt"))
random.shuffle(label_files)

for lf in label_files:
    boxes = defaultdict(list)
    for line in lf.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        counts[cid] += 1
        boxes[cid].append(list(map(float, parts[1:])))
    for cid, b in boxes.items():
        if len(examples[cid]) < SAMPLES_PER_CLASS:
            examples[cid].append((lf.stem, b))

print("\nClass ID -> number of boxes")
for cid in sorted(counts):
    print(f"  {cid}: {counts[cid]}")

out = Path("check_samples")
out.mkdir(exist_ok=True)

for cid, items in examples.items():
    for n, (stem, boxes) in enumerate(items):
        img_path = next((img_dir / f"{stem}{e}" for e in (".jpg", ".jpeg", ".png")
                         if (img_dir / f"{stem}{e}").exists()), None)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for cx, cy, bw, bh in boxes:
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"class {cid}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imwrite(str(out / f"class_{cid}_{n}.jpg"), img)

print(f"\nSaved sample images to {out}/ - open them and note what each class ID looks like.")