import os
import json
from torch.utils.data import Dataset


class HatefulMemesDataset(Dataset):
    def __init__(self, data_dir, split):
        """
        data_dir: path to folder containing img/, train.jsonl, dev.jsonl, test.jsonl
        split: one of ["train", "dev", "test"]
        """

        assert split in ["train", "dev", "test"]

        self.data_dir = data_dir
        self.split = split
        self.img_dir = os.path.join(data_dir, "img")
        self.jsonl_path = os.path.join(data_dir, f"{split}.jsonl")

        self.samples = self._load_jsonl()

    def _load_jsonl(self):
        samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())

                sample = {
                    "id": item["id"],
                    "img_path": os.path.join(self.data_dir, item["img"]),
                    "text": item["text"],
                    "label": item.get("label", -1)  # test set may not have labels
                }

                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]