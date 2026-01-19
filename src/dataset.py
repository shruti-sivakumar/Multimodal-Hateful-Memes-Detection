from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class HatefulMemesDataset(Dataset):
    """
    Returns:
      image: Tensor [3,224,224]
      text:  str
      label: Tensor []
    """
    def __init__(self, jsonl_path: str, img_dir: str, transform=None):
        self.df = pd.read_json(jsonl_path, lines=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = Path(str(row["img"]).replace("\\", "/")).name
        img_path = self.img_dir / img_name

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text = row["text"]
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return image, text, label


def collate_fn(batch):
    images, texts, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, list(texts), labels