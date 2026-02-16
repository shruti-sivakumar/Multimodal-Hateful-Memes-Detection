import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import models
from src.dataset import HatefulMemesDataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16  # ViT tokens are heavy

OUT_DIR = "artifacts/tokens/image"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    vit.heads = torch.nn.Identity()  # remove classifier head
    vit.to(DEVICE)
    vit.eval()

    transform = models.ViT_B_16_Weights.DEFAULT.transforms()

    # We need access to token embeddings.
    # Torchvision ViT exposes them via forward_features().
    # We'll use vit._process_input + encoder manually via forward_features() when available.
    # In torchvision, vit.forward_features(x) returns [B, 768] (pooled) in some versions,
    # so we use a safe hook approach on vit.encoder output.

    def extract_tokens(batch_imgs):
        """
        Returns tokens [B, 197, 768]
        """
        x = torch.stack(batch_imgs).to(DEVICE)

        with torch.no_grad():
            # process input into patch embeddings
            x = vit._process_input(x)  # [B, 196, 768]
            n = x.shape[0]

            # add class token
            cls_token = vit.class_token.expand(n, -1, -1)  # [B,1,768]
            x = torch.cat([cls_token, x], dim=1)  # [B,197,768]

            # add position embedding + dropout
            x = x + vit.encoder.pos_embedding
            x = vit.encoder.dropout(x)

            # transformer encoder
            x = vit.encoder.layers(x)
            x = vit.encoder.ln(x)  # [B,197,768]

        return x.cpu().to(torch.float16).numpy()

    for split in ["train", "dev", "test"]:
        ds = HatefulMemesDataset("data", split)

        tokens_all = []
        labels = []
        ids = []

        for i in tqdm(range(0, len(ds.samples), BATCH_SIZE), desc=f"ViT tokens {split}"):
            batch = ds.samples[i:i+BATCH_SIZE]

            batch_imgs = []
            for s in batch:
                img = Image.open(s["img_path"]).convert("RGB")
                batch_imgs.append(transform(img))

            tok = extract_tokens(batch_imgs)  # [B,197,768]
            tokens_all.append(tok)

            labels.extend([s["label"] for s in batch])
            ids.extend([s["id"] for s in batch])

        tokens_all = np.concatenate(tokens_all, axis=0)

        np.save(f"{OUT_DIR}/{split}_vit_tokens.npy", tokens_all)

        # (optional) meta saved already by text script, but harmless if you want it here too:
        os.makedirs("artifacts/tokens/meta", exist_ok=True)
        np.save(f"artifacts/tokens/meta/{split}_labels.npy", np.array(labels))
        np.save(f"artifacts/tokens/meta/{split}_ids.npy", np.array(ids))

    print("Done.")


if __name__ == "__main__":
    main()
