import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

img_dir, lbl_dir = Path(sys.argv[1]), Path(sys.argv[2])
N, TILE = 16, 200
random.seed(1)

label_files = list(lbl_dir.glob("*.txt"))
random.shuffle(label_files)

crops = defaultdict(list)
shapes = defaultdict(list)    # class -> list of (w, h) as image fractions

for lf in label_files:
    img = None
    for line in lf.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        shapes[cid].append((bw, bh))
        if len(crops[cid]) >= N:
            continue
        if img is None:
            p = next((img_dir / f"{lf.stem}{e}" for e in (".jpg", ".jpeg", ".png")
                      if (img_dir / f"{lf.stem}{e}").exists()), None)
            if p is None:
                break
            img = cv2.imread(str(p))
        h, w = img.shape[:2]
        x1, y1 = max(int((cx - bw / 2) * w), 0), max(int((cy - bh / 2) * h), 0)
        x2, y2 = min(int((cx + bw / 2) * w), w), min(int((cy + bh / 2) * h), h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops[cid].append(cv2.resize(crop, (TILE, TILE)))

print("Class | median width | median height | median w/h ratio")
for cid in sorted(shapes):
    ws = [s[0] for s in shapes[cid]]
    hs = [s[1] for s in shapes[cid]]
    ratio = statistics.median(w / h for w, h in shapes[cid] if h > 0)
    print(f"  {cid}   |    {statistics.median(ws):.3f}     |    {statistics.median(hs):.3f}      |    {ratio:.2f}")

for cid, tiles in crops.items():
    while len(tiles) < N:
        tiles.append(np.zeros((TILE, TILE, 3), dtype=np.uint8))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, N, 4)]
    cv2.imwrite(f"crops_class_{cid}.jpg", np.vstack(rows))

print("\nSaved crops_class_<id>.jpg for each class. Open them and compare.")