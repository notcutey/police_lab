# vt_siglip/data_linked.py
from dataclasses import dataclass
from typing import List, Dict, Any, Hashable, Set

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# --------- 유틸 ---------
def _to_label_set(x: Any) -> Set[Hashable]:
    if x is None: return set()
    if isinstance(x, (str, int)): return {x}
    if isinstance(x, set): return x
    return set(x)

# 이미지 멀티라벨 vs 텍스트 싱글라벨 → 타깃 행렬
def build_targets_imgmulti_textsingle(
    img_label_sets: List[Set[Hashable]],  # 길이 B, 각 원소는 set(...)
    txt_labels: List[Hashable],           # 길이 M, 각 원소는 단일 라벨
) -> torch.Tensor:
    B, M = len(img_label_sets), len(txt_labels)

    Y = torch.zeros((B, M), dtype=torch.float32)
    for i in range(B):
        Li = img_label_sets[i]
        if not Li: continue
        for j in range(M):
            if txt_labels[j] in Li:
                Y[i, j] = 1.0
    return Y

# --------- 이미지(멀티라벨) ---------
class ImageDatasetMultiLabel(Dataset):
    """
    items_img: [{"image_path": str, "labels": list|set|str|int}, ...]
    """
    def __init__(self, items_img: List[Dict[str, Any]], image_transform=None):
        self.items = items_img
        self.tfm = image_transform
        from PIL import Image
        self._open = Image.open

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = self._open(it["image_path"]).convert("RGB")
        if self.tfm is not None:
            img = self.tfm(img)
        else:
            import torchvision.transforms as T
            img = T.ToTensor()(img)
        labels = _to_label_set(it.get("labels", []))
        return {"image": img, "labels": labels}

@dataclass
class ImageBatchMulti:
    images: torch.Tensor                 # [B, 3, H, W]
    label_sets: List[Set[Hashable]]      # 길이 B

class ImageCollatorMulti:
    def __call__(self, batch: List[Dict[str, Any]]) -> ImageBatchMulti:
        images = torch.stack([b["image"] for b in batch], dim=0)
        label_sets = [b["labels"] for b in batch]
        return ImageBatchMulti(images=images, label_sets=label_sets)

# --------- 텍스트(싱글라벨) ---------
class TextDatasetSingleLabel(Dataset):
    """
    items_txt: [{"text": str, "label": str|int}, ...]
    """
    def __init__(self, items_txt: List[Dict[str, Any]]):
        self.items = items_txt

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {"text": it["text"], "label": it["label"]}

@dataclass
class TextBatchSingle:
    input_ids: torch.Tensor              # [M, L]
    attention_mask: torch.Tensor         # [M, L]
    texts: List[str]                     # 길이 M
    labels: List[Hashable]               # 길이 M (싱글라벨)

class TextCollatorSingle:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 128):
        self.tok = tokenizer
        self.max_len = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> TextBatchSingle:
        texts = [b["text"] for b in batch]
        labels = [b["label"] for b in batch]
        tok_out = self.tok(
            texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return TextBatchSingle(
            input_ids=tok_out["input_ids"],
            attention_mask=tok_out["attention_mask"],
            texts=texts,
            labels=labels,
        )
