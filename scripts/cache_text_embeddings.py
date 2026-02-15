import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from src.dataset import HatefulMemesDataset


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TEXT_MODELS = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}

MAX_LEN = 64
BATCH_SIZE = 32


def get_embeddings(data_dir, split, model_name, hf_name):

    dataset = HatefulMemesDataset(data_dir, split)

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name)
    model.to(DEVICE)
    model.eval()

    all_embeddings = []
    all_labels = []
    all_ids = []

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):

        batch_samples = dataset.samples[i:i+BATCH_SIZE]
        texts = [s["text"] for s in batch_samples]
        labels = [s["label"] for s in batch_samples]
        ids = [s["id"] for s in batch_samples]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token
            pooled = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(pooled.cpu())
        all_labels.extend(labels)
        all_ids.extend(ids)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()

    return embeddings, np.array(all_labels), np.array(all_ids)


def main():

    data_dir = "data"
    os.makedirs("artifacts/embeddings/text", exist_ok=True)

    for split in ["train", "dev", "test"]:
        split_labels = None
        split_ids = None

        for model_key, hf_name in TEXT_MODELS.items():
            print(f"\nCaching {model_key} - {split}")
            emb, labels, ids = get_embeddings(data_dir, split, model_key, hf_name)
            np.save(f"artifacts/embeddings/text/{split}_{model_key}.npy", emb)
            if split_labels is None:
                split_labels = labels
                split_ids = ids

        np.save(f"artifacts/embeddings/text/{split}_labels.npy", split_labels)
        np.save(f"artifacts/embeddings/text/{split}_ids.npy", split_ids)

    print("Done.")


if __name__ == "__main__":
    main()
