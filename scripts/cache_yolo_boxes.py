import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from src.dataset import HatefulMemesDataset

YOLO_MODEL_PATH = "models/yolov8n.pt"
CONF = 0.25
IOU = 0.45
MAX_DET = 5

def run_split(data_dir: str, split: str, out_path: str):
    ds = HatefulMemesDataset(data_dir, split)
    yolo = YOLO(YOLO_MODEL_PATH)

    # map: id -> list of boxes [x1,y1,x2,y2]
    boxes_map = {}

    for s in tqdm(ds.samples, desc=f"YOLO boxes {split}"):
        img = Image.open(s["img_path"]).convert("RGB")

        r = yolo(img, verbose=False, conf=CONF, iou=IOU, max_det=MAX_DET)[0]
        if r.boxes is None or len(r.boxes) == 0:
            boxes = []
        else:
            boxes = []
            for b in r.boxes.xyxy.tolist():
                x1, y1, x2, y2 = [int(v) for v in b]
                boxes.append([x1, y1, x2, y2])

        boxes_map[str(s["id"])] = boxes

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "yolo_model": YOLO_MODEL_PATH,
                "conf": CONF,
                "iou": IOU,
                "max_det": MAX_DET,
                "split": split,
                "boxes": boxes_map,
            },
            f
        )

def main():
    data_dir = "data"
    os.makedirs("artifacts/yolo_boxes", exist_ok=True)

    for split in ["train", "dev", "test"]:
        out_path = f"artifacts/yolo_boxes/{split}.json"
        run_split(data_dir, split, out_path)

    print("Done.")

if __name__ == "__main__":
    main()
