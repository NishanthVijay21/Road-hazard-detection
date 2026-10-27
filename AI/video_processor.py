import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IGNORE_CLASSES = {"marking_or_manhole"}   # detected by the model, but not a road hazard

# COCO class ids used by the privacy model
COCO_PERSON = 0
COCO_VEHICLES = [2, 3, 5, 7]              # car, motorcycle, bus, truck


# ----------------------------- GPS helpers -----------------------------

def load_gps(csv_path):
    """Load the optional GPS sidecar file. Returns None if no file is given."""
    if not csv_path:
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"GPS file {csv_path} has no rows")

    def col(name):
        return np.array([float(r[name]) if r.get(name) not in (None, "") else np.nan
                         for r in rows])

    t = col("time_s")
    order = np.argsort(t)
    return {
        "t": t[order],
        "lat": col("latitude")[order],
        "lon": col("longitude")[order],
        "heading": col("heading")[order],
        "speed": col("speed")[order],
    }


def _clean(x):
    return None if x is None or np.isnan(x) else float(x)


def gps_at(gps, t_sec):
    """Interpolated position at a video time. Everything is None when there is no GPS file."""
    if gps is None:
        return {"latitude": None, "longitude": None, "heading": None, "speed": None}
    nearest = int(np.abs(gps["t"] - t_sec).argmin())
    return {
        "latitude": _clean(np.interp(t_sec, gps["t"], gps["lat"])),
        "longitude": _clean(np.interp(t_sec, gps["t"], gps["lon"])),
        "heading": _clean(gps["heading"][nearest]),
        "speed": _clean(gps["speed"][nearest]),
    }


# ----------------------------- privacy blur ------------------------------

def blur_region(frame, x1, y1, x2, y2):
    """Pixelate + blur one rectangle of the frame, in place. Coordinates are clipped to the frame."""
    h, w = frame.shape[:2]
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), w), min(int(y2), h)
    rw, rh = x2 - x1, y2 - y1
    if rw < 2 or rh < 2:
        return False
    roi = frame[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, rw // 10), max(1, rh // 10)), interpolation=cv2.INTER_LINEAR)
    block = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(block, (0, 0), sigmaX=max(rw, rh) / 20 + 1)
    return True


def privacy_zones(boxes_xyxy, classes):
    """Turn person/vehicle boxes into the rectangles that must be blurred."""
    zones = []
    for (x1, y1, x2, y2), cls in zip(boxes_xyxy, classes):
        bw, bh = x2 - x1, y2 - y1
        if cls == COCO_PERSON:      # head and face area: top 35% of the person, a bit wider
            zones.append((x1 - 0.1 * bw, y1 - 0.05 * bh, x2 + 0.1 * bw, y1 + 0.35 * bh))
        else:                       # vehicle: lower half, where number plates are
            zones.append((x1, y1 + 0.5 * bh, x2, y2))
    return zones


class PrivacyBlur:
    """Finds people and vehicles with a small COCO model and blurs the sensitive zones."""

    def __init__(self, weights, device, conf):
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.regions_blurred = 0

    def apply(self, frame):
        """Blur in place and return the frame."""
        res = self.model.predict(
            frame, conf=self.conf, imgsz=1280, device=self.device,
            classes=[COCO_PERSON] + COCO_VEHICLES, verbose=False,
        )[0]
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.int().cpu().tolist()
            for zone in privacy_zones(boxes, classes):
                self.regions_blurred += blur_region(frame, *zone)
        return frame


# ------------------------- image similarity features ---------------------

class Embedder:
    """512-number feature vector per crop (ResNet-18), L2-normalised.
    Similarity of two crops = dot product of their vectors (1.0 = identical)."""

    def __init__(self, device):
        import torch
        import torchvision
        self.torch = torch
        self.dev = torch.device(f"cuda:{device}" if isinstance(device, int) else "cpu")
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(self.dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, bgr):
        torch = self.torch
        rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
        x = ((x - self.mean) / self.std).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            v = self.net(x)[0]
            v = v / v.norm()
        return [round(float(a), 4) for a in v.cpu()]


# --------------------------- evidence crops ------------------------------

def render_crops(video_path, kept, privacy, width, height):
    """Second pass: re-read each hazard's best frame, blur it, then cut out the crop."""
    needed = {}
    for tr in kept:
        needed.setdefault(tr["frame"], []).append(tr)
    if not needed:
        return

    cap = cv2.VideoCapture(video_path)
    last = max(needed)
    idx = -1
    while idx < last:
        if not cap.grab():
            break
        idx += 1
        if idx not in needed:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        if privacy is not None:
            frame = privacy.apply(frame)          # blur BEFORE cutting the crop
        for tr in needed[idx]:
            x1, y1, x2, y2 = tr["bbox"]
            pad_x, pad_y = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            cx1, cy1 = max(int(x1 - pad_x), 0), max(int(y1 - pad_y), 0)
            cx2, cy2 = min(int(x2 + pad_x), width), min(int(y2 + pad_y), height)
            tr["crop"] = frame[cy1:cy2, cx1:cx2].copy()
    cap.release()


# ---------------------------- main processing ---------------------------

def process_video(args):
    model = YOLO(args.weights)
    keep_ids = [i for i, name in model.names.items() if name not in IGNORE_CLASSES]
    print("Hazard classes:", {i: model.names[i] for i in keep_ids})

    privacy = None if args.no_blur else PrivacyBlur(args.privacy_weights, args.device, args.privacy_conf)
    embedder = Embedder(args.device) if args.embed else None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = (datetime.fromisoformat(args.start_time) if args.start_time
             else datetime.now().replace(microsecond=0))
    gps = load_gps(args.gps)

    out_dir = Path(args.out)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / args.stride, (width, height))

    tracks = {}      # track_id -> best observation so far (no images are kept in memory)
    frame_idx = -1

    # ---- Pass 1: detect + track on the original frames ----
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.stride:
            continue

        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=args.conf, imgsz=args.imgsz, device=args.device,
            classes=keep_ids, verbose=False,
        )[0]

        if writer is not None:
            vis = result.plot()
            if privacy is not None and args.blur_video:
                vis = privacy.apply(vis)
            writer.write(vis)

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()

            for tid, conf, cls, (x1, y1, x2, y2) in zip(ids, confs, clss, xyxy):
                tr = tracks.setdefault(tid, {"frames": 0, "conf": 0.0})
                tr["frames"] += 1
                if conf > tr["conf"]:                 # remember the most confident view
                    tr.update(
                        conf=float(conf),
                        hazard_type=model.names[cls],
                        frame=frame_idx,
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                    )

        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total}  |  tracks so far: {len(tracks)}")

    cap.release()
    if writer is not None:
        writer.release()

    # Drop very short tracks (flicker / false alarms)
    kept = [t for t in tracks.values() if t["frames"] >= args.min_frames and "bbox" in t]
    kept.sort(key=lambda t: t["frame"])

    # ---- Pass 2: blurred evidence crops ----
    print("Creating evidence crops" + (" with privacy blur..." if privacy else " (privacy blur OFF)..."))
    render_crops(args.video, kept, privacy, width, height)
    kept = [t for t in kept if t.get("crop") is not None and t["crop"].size > 0]

    records = []
    for n, tr in enumerate(kept, start=1):
        name = f"hazard_{n:03d}.jpg"
        cv2.imwrite(str(crops_dir / name), tr["crop"])
        t_sec = tr["frame"] / fps
        record = {
            "hazard_type": tr["hazard_type"],
            "confidence": round(tr["conf"], 3),
            "bbox": tr["bbox"],                       # pixels, original frame size
            "image_path": name,                       # file inside <out>/crops/
            "timestamp": (start + timedelta(seconds=t_sec)).isoformat(),
            **gps_at(gps, t_sec),
        }
        if embedder is not None:                      # computed on the BLURRED crop
            record["embedding"] = embedder(tr["crop"])
        records.append(record)

    json_path = out_dir / "detections.json"
    json_path.write_text(json.dumps(records, indent=2))

    # Summary
    print(f"\nProcessed {frame_idx + 1} frames -> {len(tracks)} tracks -> {len(records)} detections "
          f"(dropped {len(tracks) - len(records)} short tracks)")
    counts = {}
    for r in records:
        counts[r["hazard_type"]] = counts.get(r["hazard_type"], 0) + 1
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"Saved: {json_path}  and crops in {crops_dir}")

    if privacy is not None:
        print(f"Privacy: {privacy.regions_blurred} regions (heads / vehicle plate areas) blurred. "
              f"Check the crops by eye before sharing them.")
        if args.save_video and not args.blur_video:
            print("Note: annotated.mp4 is NOT blurred. Add --blur-video if you will show or share it.")
    else:
        print("WARNING: privacy blur was switched off, crops may show faces and number plates.")
    if embedder is not None:
        print("Similarity embeddings added (512 numbers per record, field 'embedding').")

    if gps is None:
        print("Note: no GPS file given, so latitude/longitude/heading/speed are null.")
    elif kept and kept[-1]["frame"] / fps > gps["t"][-1] + 0.5:
        print(f"WARNING: the video is {total / fps:.0f} s long but the GPS file ends at "
              f"{gps['t'][-1]:.0f} s. Detections after that time all get the LAST GPS position. "
              f"Extend the GPS file to cover the whole video.")

    if args.api_url:
        post_to_api(records, args.api_url)


def post_to_api(records, url):
    """Optional: send each record to the backend (POST /detections).
    Only the JSON fields are sent; agree with Developer 2 how the crop image is transferred."""
    import requests
    ok = 0
    for r in records:
        try:
            resp = requests.post(url, json=r, timeout=10)
            ok += resp.ok
        except requests.RequestException as e:
            print("API error:", e)
            break
    print(f"Posted {ok}/{len(records)} records to {url}")


def parse_args():
    ap = argparse.ArgumentParser(description="Detect road hazards in a video.")
    ap.add_argument("--video", required=True, help="path to the input video")
    ap.add_argument("--weights", default="models/road_hazard_v2.pt")
    ap.add_argument("--out", default="output", help="output folder (use a new one per video)")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=800, help="use the same size you trained with")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (2-3 = faster)")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="a hazard must be tracked in at least this many frames")
    ap.add_argument("--device", default=None, help="'0' for the first GPU or 'cpu' (auto if empty)")
    ap.add_argument("--gps", default=None, help="optional GPS CSV (see top of file)")
    ap.add_argument("--start-time", default=None,
                    help="ISO time of video start, e.g. 2026-10-26T10:12:00 (default: now)")
    ap.add_argument("--save-video", action="store_true", help="also write annotated.mp4")
    ap.add_argument("--api-url", default=None, help="optional, e.g. http://localhost:8000/detections")

    # privacy and similarity features
    ap.add_argument("--no-blur", action="store_true", help="switch privacy blur OFF (not recommended)")
    ap.add_argument("--blur-video", action="store_true", help="also blur annotated.mp4 (slower)")
    ap.add_argument("--privacy-weights", default="yolov8n.pt", help="COCO model used to find people/vehicles")
    ap.add_argument("--privacy-conf", type=float, default=0.25,
                    help="lower = blur more aggressively (catches more people, more over-blurring)")
    ap.add_argument("--embed", action="store_true", help="add an image-similarity 'embedding' to each record")
    args = ap.parse_args()

    if args.device is None:
        import torch
        args.device = 0 if torch.cuda.is_available() else "cpu"
    elif args.device.isdigit():
        args.device = int(args.device)
    return args


if __name__ == "__main__":
    process_video(parse_args())