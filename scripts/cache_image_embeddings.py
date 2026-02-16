import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import models
from src.dataset import HatefulMemesDataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

IMAGE_MODELS = ["resnet50", "vgg16", "vit"]

def get_model_and_transform(model_name: str):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # [B,2048,1,1]
        transform = models.ResNet50_Weights.DEFAULT.transforms()
        out_dim = 2048

    elif model_name == "vgg16":
        base = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Keep feature extractor + avgpool, and keep classifier up to fc7 (4096)
        model = torch.nn.Sequential(
            base.features,
            base.avgpool,
            torch.nn.Flatten(),
            *list(base.classifier.children())[:-1]  # drops final Linear(4096->1000)
        )
        transform = models.VGG16_Weights.DEFAULT.transforms()
        out_dim = 4096

    elif model_name == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads = torch.nn.Identity()        # [B,768]
        transform = models.ViT_B_16_Weights.DEFAULT.transforms()
        out_dim = 768

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(DEVICE)
    model.eval()
    return model, transform, out_dim

def embed_one_image(model, transform, pil_img, model_name: str):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = model(x)

    # resnet: [1,2048,1,1] -> [1,2048]
    if model_name == "resnet50" and y.dim() == 4:
        y = y.view(y.size(0), -1)
    return y.squeeze(0).cpu()  # [D]

def load_boxes(split: str):
    path = f"artifacts/yolo_boxes/{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["boxes"]  # dict str(id) -> list[[x1,y1,x2,y2],...]

def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return (x1, y1, x2, y2)

def main():
    data_dir = "data"
    os.makedirs("artifacts/embeddings/image", exist_ok=True)

    for split in ["train", "dev", "test"]:
        ds = HatefulMemesDataset(data_dir, split)
        boxes_map = load_boxes(split)

        # labels/ids saved once per split
        labels = np.array([s["label"] for s in ds.samples])
        ids = np.array([s["id"] for s in ds.samples])

        np.save(f"artifacts/embeddings/image/{split}_labels.npy", labels)
        np.save(f"artifacts/embeddings/image/{split}_ids.npy", ids)

        for model_name in IMAGE_MODELS:
            print(f"\nCaching {model_name} (global + yolo) - {split}")

            model, transform, out_dim = get_model_and_transform(model_name)

            global_embs = np.zeros((len(ds), out_dim), dtype=np.float32)
            yolo_embs   = np.zeros((len(ds), out_dim), dtype=np.float32)

            for i, s in enumerate(tqdm(ds.samples, desc=f"{split}-{model_name}")):
                img = Image.open(s["img_path"]).convert("RGB")
                w, h = img.size

                # Global embedding
                g = embed_one_image(model, transform, img, model_name)
                global_embs[i] = g.numpy()

                # YOLO boxes (cached)
                boxes = boxes_map.get(str(s["id"]), [])
                obj_embs = []

                for b in boxes:
                    cb = clamp_box(b, w, h)
                    if cb is None:
                        continue
                    crop = img.crop(cb)
                    e = embed_one_image(model, transform, crop, model_name)
                    obj_embs.append(e)

                if len(obj_embs) == 0:
                    # fallback to global
                    yolo_embs[i] = global_embs[i]
                else:
                    yolo_embs[i] = torch.stack(obj_embs).mean(dim=0).numpy()

            np.save(f"artifacts/embeddings/image/{split}_{model_name}.npy", global_embs)
            np.save(f"artifacts/embeddings/image/{split}_{model_name}_yolo.npy", yolo_embs)

    print("Done.")

if __name__ == "__main__":
    main()
