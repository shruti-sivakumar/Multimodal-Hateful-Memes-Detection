import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from src.dataset import HatefulMemesDataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TEXT_MODEL = "roberta-base"
MAX_LEN = 64
BATCH_SIZE = 64  # safe on Mac

OUT_DIR = "artifacts/tokens/text"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL)
    model.to(DEVICE)
    model.eval()

    for split in ["train", "dev", "test"]:
        ds = HatefulMemesDataset("data", split)

        tokens_all = []
        masks_all = []
        labels = []
        ids = []

        # batch over samples
        for i in tqdm(range(0, len(ds.samples), BATCH_SIZE), desc=f"Text tokens {split}"):
            batch = ds.samples[i:i+BATCH_SIZE]
            texts = [s["text"] for s in batch]

            enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                h = out.last_hidden_state  # [B, L, 768]

            # store float16 on disk
            tokens_all.append(h.cpu().to(torch.float16).numpy())
            masks_all.append(attention_mask.cpu().numpy().astype(np.int8))

            labels.extend([s["label"] for s in batch])
            ids.extend([s["id"] for s in batch])

        tokens_all = np.concatenate(tokens_all, axis=0)  # [N, L, 768]
        masks_all = np.concatenate(masks_all, axis=0)    # [N, L]

        np.save(f"{OUT_DIR}/{split}_roberta_tokens.npy", tokens_all)
        np.save(f"{OUT_DIR}/{split}_roberta_mask.npy", masks_all)

        # save ids/labels once (useful later)
        os.makedirs("artifacts/tokens/meta", exist_ok=True)
        np.save(f"artifacts/tokens/meta/{split}_labels.npy", np.array(labels))
        np.save(f"artifacts/tokens/meta/{split}_ids.npy", np.array(ids))

    print("Done.")


if __name__ == "__main__":
    main()