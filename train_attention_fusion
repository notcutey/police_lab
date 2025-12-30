#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import json
import time
import random
from typing import Dict, Any, Tuple, List, Optional
import argparse
import glob
from collections import defaultdict
import re
import os.path as osp
from types import SimpleNamespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

from llm_machine import LLMTextEncoder
from llm_machine import VisionTextSigLIP
from llm_machine.data_linked import ImageDatasetMultiLabel, ImageCollatorMulti, TextCollatorSingle
from networks import Token
from llm_machine import train_step_linked
from llm_machine import log_print

# ===================== #
#   기본 설정 / 경로     #
# ===================== #

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

image_size = (1024, 512)
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD  = [0.26862954, 0.26130258, 0.27577711]

ITEMS_IMG_PATH = "/root/project/llm_prompt_new/json_file/image_paths.jsonl"
ITEMS_TXT_PATH = "/root/project/llm_prompt_new/json_file/최종_final_txt_pattern.jsonl"

TOKEN_CKPT_PATH = None

CKPT_DIR = "/root/project/llm_prompt/llm_machine/checkpoint_siglip"
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS = 600
BATCH_SIZE_IMG = 64
NUM_WORKERS_IMG = 4

LR = 1e-5
WEIGHT_DECAY = 0.01

LABEL_HIT_AT_K = 5
LOG_EVERY_STEPS = 10

DEBUG_BACKPROP = True
DEBUG_SAMPLE_PARAMS = 3
DEBUG_EVERY_STEPS = 1

# ===================== #
#     유틸 함수 묶음     #
# ===================== #

import psutil
from contextlib import contextmanager

def _now():
    return time.strftime("%H:%M:%S")

def _fmt_eta(sec: float) -> str:
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def _mem_gb() -> str:
    if torch.cuda.is_available():
        mb = torch.cuda.max_memory_allocated() / (1024**2)
        return f"GPU max {mb:,.0f} MB"
    else:
        return f"RAM used {psutil.Process().memory_info().rss / (1024**3):.2f} GB"

def log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)

@contextmanager
def timeit(tag: str):
    t0 = time.perf_counter()
    log(f"▶ {tag} ...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log(f"✔ {tag} done in {_fmt_eta(dt)}  ({_mem_gb()})")

def _safe_torch_load(ckpt_path: str, map_location="cpu"):
    try:
        import numpy as np
        from torch.serialization import safe_globals
        with safe_globals([np.core.multiarray.scalar]):
            try:
                return torch.load(ckpt_path, map_location=map_location)
            except TypeError:
                return torch.load(ckpt_path, map_location=map_location)
    except Exception:
        try:
            return torch.load(ckpt_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location=map_location)

def _unwrap_state_dict(maybe_wrapped: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("state_dict", "model", "net", "module"):
        if key in maybe_wrapped and isinstance(maybe_wrapped[key], dict):
            return maybe_wrapped[key]
    return maybe_wrapped

def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def _select_by_prefix_and_shape(
    checkpoint: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    allowed_prefixes: Tuple[str, ...] = ("backbone", "tr"),
) -> Dict[str, torch.Tensor]:
    filtered = {}
    for k, v in checkpoint.items():
        if not any(k.startswith(pfx) for pfx in allowed_prefixes):
            continue
        if (k in model_state) and (model_state[k].shape == v.shape):
            filtered[k] = v
    return filtered

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

# ===================== #
#   분리 로드/세이브 유틸  #
# ===================== #

def save_weights(vt, token_model, optimizer, scaler, epoch, global_step, tag="last", save_full_state=True):
    """vt/token 개별 저장 (+ 옵션: 통합 상태 저장)"""
    vt_sd = vt.state_dict()
    token_sd = (token_model.module if hasattr(token_model, "module") else token_model).state_dict()

    # ▶ 개별 가중치 파일 이름 (infer 기본값과 일치)
    vt_pth_path = os.path.join(CKPT_DIR, f"vt_{tag}_final_self_attention_multimodal_1.pth")
    token_pth_path = os.path.join(CKPT_DIR, f"token_{tag}_final_self_attention_multimodal_1.pth")

    torch.save({"state_dict": vt_sd}, vt_pth_path)
    torch.save({"state_dict": token_sd}, token_pth_path)
    print(f"[Save] vt (wrapped)   -> {vt_pth_path}")
    print(f"[Save] token (wrapped)-> {token_pth_path}")

    # 필요하면 통합 train_state 저장 (resume용)
    if save_full_state:
        full_path = os.path.join(CKPT_DIR, f"train_state_{tag}.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "vt": vt_sd,
                "token": token_sd,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
            full_path,
        )
        print(f"[Save] full ckpt -> {full_path}")

def load_vt_from_path(vt: torch.nn.Module, path: str, map_location="cpu", strict: bool = False):
    assert os.path.isfile(path), f"vt ckpt not found: {path}"
    raw = _safe_torch_load(path, map_location=map_location)

    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        sd = raw
    else:
        sd = _unwrap_state_dict(raw)
    sd = _strip_prefix_if_present(sd, "module.")

    model_state = vt.state_dict()
    ckpt_keys = set(sd.keys())
    model_keys = set(model_state.keys())

    missing_in_ckpt   = sorted(model_keys - ckpt_keys)
    unexpected_in_ckpt = sorted(ckpt_keys - model_keys)

    print(f"[LoadVT][DEBUG] #ckpt_keys={len(ckpt_keys)} | #model_keys={len(model_keys)}")
    if missing_in_ckpt:
        print(f"[LoadVT][DEBUG] Missing keys in ckpt (first 30):")
        for k in missing_in_ckpt[:30]:
            print("   (MISSING)", k)
    if unexpected_in_ckpt:
        print(f"[LoadVT][DEBUG] Unexpected keys in ckpt (first 30):")
        for k in unexpected_in_ckpt[:30]:
            print("   (UNEXPECTED)", k)

    interesting_substr = ["attn", "cross", "text_emb", "prototype", "adapter", "lora"]
    print("[LoadVT][DEBUG] Missing keys (filtered by attn/cross/text_emb/prototype/adapter/lora):")
    for k in missing_in_ckpt:
        if any(s in k for s in interesting_substr):
            print("   (MISSING-ATTN)", k)

    incompatible = vt.load_state_dict(sd, strict=strict)
    try:
        miss = list(getattr(incompatible, "missing_keys", []))
        unexp = list(getattr(incompatible, "unexpected_keys", []))
        if miss:
            print(f"[LoadVT] load_state_dict missing_keys (first 30):")
            for k in miss[:30]:
                print("   (MISS)", k)
        if unexp:
            print(f"[LoadVT] load_state_dict unexpected_keys (first 30):")
            for k in unexp[:30]:
                print("   (UNEXP)", k)
    except Exception:
        pass

    print(f"[LoadVT] loaded vt weights from {path} (strict={strict})")

def load_token_from_path(token_model: torch.nn.Module, path: str, map_location="cpu", strict: bool = False):
    assert os.path.isfile(path), f"token ckpt not found: {path}"
    raw = _safe_torch_load(path, map_location=map_location)
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        sd = raw
    else:
        sd = _unwrap_state_dict(raw)
    sd = _strip_prefix_if_present(sd, "module.")

    model_wo_ddp = token_model.module if hasattr(token_model, "module") else token_model
    model_dict = model_wo_ddp.state_dict()
    filtered = _select_by_prefix_and_shape(sd, model_dict, allowed_prefixes=("backbone", "tr"))
    if len(filtered) == 0:
        filtered = {
            k: v for k, v in sd.items()
            if (k in model_dict) and (model_dict[k].shape == v.shape)
            and not any(x in k for x in ["classifier", "fc", "head", "heads", "arcface"])
        }
    incompatible = model_wo_ddp.load_state_dict(filtered, strict=False if not strict else True)
    miss = list(getattr(incompatible, "missing_keys", []))
    unexp = list(getattr(incompatible, "unexpected_keys", []))
    print(f"[LoadToken] loaded {len(filtered)} keys from {path}")
    if miss:
        print(f"[LoadToken] Missing keys: {len(miss)} (first 10): {miss[:10]}")
    if unexp:
        print(f"[LoadToken] Unexpected keys: {len(unexp)} (first 10): {unexp[:10]}")

def load_train_state_or_pair(vt, token_model, tag: str = "last") -> Optional[int]:
    """
    - train_state_{tag}.pt 가 있으면 거기서 epoch/optimizer까지 로드
    - 없으면 vt_{tag}_final_cross_attention_multimodal.pth / token_{tag}_final_cross_attention_multimodal.pth 로 시도
    """
    state_path = os.path.join(CKPT_DIR, f"train_state_{tag}.pt")
    vt_path    = os.path.join(CKPT_DIR, f"vt_{tag}_final_cross_attention_multimodal.pth")
    token_path = os.path.join(CKPT_DIR, f"token_{tag}_final_cross_attention_multimodal.pth")

    epoch = None
    if os.path.isfile(state_path):
        raw = _safe_torch_load(state_path, map_location="cpu")
        if "vt" in raw:
            vt.load_state_dict(raw["vt"], strict=False)
        if "token" in raw:
            (token_model.module if hasattr(token_model, "module") else token_model).load_state_dict(
                raw["token"], strict=False
            )
        epoch = int(raw.get("epoch", 0))
        print(f"[Load] full train_state from {state_path} (epoch={epoch})")
        return epoch

    ok = False
    if os.path.isfile(vt_path):
        raw_vt = _safe_torch_load(vt_path, map_location="cpu")
        vt_sd = _unwrap_state_dict(raw_vt)
        vt_sd = _strip_prefix_if_present(vt_sd, "module.")
        vt.load_state_dict(vt_sd, strict=False)
        ok = True
    if os.path.isfile(token_path):
        raw_tok = _safe_torch_load(token_path, map_location="cpu")
        tok_sd = _unwrap_state_dict(raw_tok)
        tok_sd = _strip_prefix_if_present(tok_sd, "module.")
        (token_model.module if hasattr(token_model, "module") else token_model).load_state_dict(
            tok_sd, strict=False
        )
        ok = True

    if ok:
        print(f"[Load] vt/token from pair files: {vt_path} , {token_path}")
    else:
        print(f"[Load] No checkpoint found for tag='{tag}' under {CKPT_DIR}")
    return epoch

def _load_resume_checkpoint(
    vt,
    token_model,
    optimizer,
    scaler,
    resume_path: Optional[str],
    resume_tag: Optional[str],
    device: str,
    resume_all: bool = False
) -> Tuple[int, int]:
    epoch = 0
    global_step = 0
    if resume_path is not None and len(str(resume_path)) > 0:
        assert os.path.isfile(resume_path), f"resume_path not found: {resume_path}"
        print(f"[Resume] Loading checkpoint from file: {resume_path}")
        raw = _safe_torch_load(resume_path, map_location=device)

        if "vt" in raw:
            vt.load_state_dict(raw["vt"], strict=False)
        if "token" in raw:
            (token_model.module if hasattr(token_model, "module") else token_model).load_state_dict(
                raw["token"], strict=False
            )

        if resume_all:
            if "optimizer" in raw and raw["optimizer"] is not None:
                optimizer.load_state_dict(raw["optimizer"])
            if "scaler" in raw and raw["scaler"] is not None and scaler is not None:
                scaler.load_state_dict(raw["scaler"])

        epoch = int(raw.get("epoch", 0))
        global_step = int(raw.get("global_step", 0))
        print(f"[Resume] Loaded: epoch={epoch}, global_step={global_step}, resume_all={resume_all}")
        return epoch + 1, global_step

    if resume_tag is not None and len(str(resume_tag)) > 0:
        state_path = os.path.join(CKPT_DIR, f"train_state_{resume_tag}.pt")
        if os.path.isfile(state_path):
            print(f"[Resume] Loading checkpoint by tag: {state_path}")
            raw = _safe_torch_load(state_path, map_location=device)

            if "vt" in raw:
                vt.load_state_dict(raw["vt"], strict=False)
            if "token" in raw:
                (token_model.module if hasattr(token_model, "module") else token_model).load_state_dict(
                    raw["token"], strict=False
                )

            if resume_all:
                if "optimizer" in raw and raw["optimizer"] is not None:
                    optimizer.load_state_dict(raw["optimizer"])
                if "scaler" in raw and raw["scaler"] is not None and scaler is not None:
                    scaler.load_state_dict(raw["scaler"])

            epoch = int(raw.get("epoch", 0))
            global_step = int(raw.get("global_step", 0))
            print(f"[Resume] Loaded: epoch={epoch}, global_step={global_step}, resume_all={resume_all}")
            return epoch + 1, global_step
        else:
            print(f"[Resume] No checkpoint for tag='{resume_tag}' under {CKPT_DIR}")
    return 1, 0

def _gather_paths(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    parts: List[str] = []
    for token in arg.replace(",", " ").split():
        if any(ch in token for ch in "*?[]"):
            parts.extend(glob.glob(token))
        else:
            parts.append(token)
    out = []
    seen = set()
    for p in parts:
        p = os.path.abspath(p)
        if (p not in seen) and os.path.isfile(p):
            out.append(p); seen.add(p)
    return out

# ---------------------
# 라벨별 텍스트 준비
# ---------------------

def build_label2texts(items_txt: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    table: Dict[int, List[str]] = defaultdict(list)
    for it in items_txt:
        lab = int(it["label"])
        table[lab].append(it["text"])
    return table

def _unique_labels_in_batch(batch_img) -> List[int]:
    if hasattr(batch_img, "label_sets"):
        uniq = set()
        for labs in batch_img.label_sets:
            for lab in labs:
                uniq.add(int(lab))
        return sorted(uniq)
    else:
        if isinstance(batch_img, dict) and "labels" in batch_img:
            labels = batch_img["labels"]
            if torch.is_tensor(labels) and labels.dim() == 2:
                uniq = set()
                for i in range(labels.size(0)):
                    idxs = labels[i].nonzero(as_tuple=False).squeeze(1).tolist()
                    uniq.update(map(int, idxs))
                return sorted(uniq)
            else:
                uniq = set()
                for labs in labels:
                    uniq.update(map(int, labs))
                return sorted(uniq)
    raise TypeError("batch_img에서 라벨 정보를 찾을 수 없다. (label_sets 또는 labels 필요)")

def _build_text_batch_one_per_label(
    batch_img,
    label2texts: Dict[int, List[str]],
    tokenizer,
    max_length: int = 128,
):
    uniq_labels = _unique_labels_in_batch(batch_img)
    items_for_collate: List[Dict[str, Any]] = []
    label_ids_in_text_order: List[int] = []
    for lab in uniq_labels:
        cand = label2texts.get(lab, [])
        if not cand:
            continue
        txt = random.choice(cand)
        items_for_collate.append({"text": txt, "label": int(lab)})
        label_ids_in_text_order.append(int(lab))

    if len(items_for_collate) == 0:
        return None

    collate = TextCollatorSingle(tokenizer, max_length=max_length)
    batch_txt = collate(items_for_collate)
    setattr(batch_txt, "label_ids", label_ids_in_text_order)
    return batch_txt

# ===================== #
#    역전파 계측 유틸     #
# ===================== #

class GradProbe:
    def __init__(self, model: torch.nn.Module, name_prefix: str = "token"):
        self.model = model
        self.name_prefix = name_prefix
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.reset()

    def attach(self):
        self.detach()
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            handle = p.register_hook(self._make_hook(name))
            self._handles.append(handle)

    def detach(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def reset(self):
        self.had_any_grad = False
        self.grad_param_count = 0
        self.total_grad_norm_sq = 0.0

    def _make_hook(self, name: str):
        def _hook(grad: torch.Tensor):
            if grad is None:
                return
            self.had_any_grad = True
            self.grad_param_count += 1
            with torch.no_grad():
                self.total_grad_norm_sq += float(grad.detach().pow(2).sum().cpu().item())
        return _hook

    def summary(self) -> Dict[str, Any]:
        total_grad_norm = (self.total_grad_norm_sq ** 0.5)
        return {
            "had_any_grad": bool(self.had_any_grad),
            "grad_param_count": int(self.grad_param_count),
            "grad_total_norm": float(total_grad_norm),
        }

class ParamSnapshot:
    def __init__(self, model: torch.nn.Module, max_params: int = 3):
        self.model = model
        self.max_params = max_params
        self.before: List[Tuple[str, torch.Tensor]] = []
        self.after:  List[Tuple[str, torch.Tensor]] = []

    def take_before(self):
        self.before = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.data.numel() >= 1024:
                self.before.append((name, p.data.detach().cpu().flatten()[:32].clone()))
                if len(self.before) >= self.max_params:
                    break

    def take_after(self):
        self.after = []
        named = dict(self.model.named_parameters())
        for name, _snap in self.before:
            p = named[name]
            self.after.append((name, p.data.detach().cpu().flatten()[:32].clone()))

    def changed_flags(self) -> List[bool]:
        flags = []
        for (n1, b), (n2, a) in zip(self.before, self.after):
            flags.append(bool(not torch.equal(b, a)))
        return flags

# ---------- 멀티라벨 메트릭: 라벨 히트 비율(R@K) ----------

def _extract_image_label_sets(batch_img) -> List[set]:
    if hasattr(batch_img, "label_sets"):
        return [set(map(int, labs)) for labs in batch_img.label_sets]

    if isinstance(batch_img, dict) and "labels" in batch_img:
        labels = batch_img["labels"]
        if torch.is_tensor(labels):
            if labels.dim() == 2:
                out = []
                for i in range(labels.size(0)):
                    idxs = labels[i].nonzero(as_tuple=False).squeeze(1).tolist()
                    out.append(set(map(int, idxs)))
                return out
            else:
                raise TypeError("labels 텐서는 [B, C] multi-hot 형태여야 한다.")
        else:
            return [set(map(int, labs)) for labs in labels]

    raise TypeError("batch_img에서 라벨 정보를 찾을 수 없다. (label_sets 또는 labels 필요)")

@torch.no_grad()
def _topk_unique_by_label(
    scores_1d,
    labels_of_texts: List[int],
    k: int,
    min_score: Optional[float] = None,
):
    if torch.is_tensor(scores_1d):
        scores_np = scores_1d.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores_1d, dtype=np.float32)

    order = np.argsort(-scores_np)
    chosen_idx, chosen_scores = [], []
    seen_labels = set()

    for j in order:
        s = float(scores_np[j])
        if (min_score is not None) and (s < min_score):
            break
        lab = int(labels_of_texts[j])
        if lab in seen_labels:
            continue
        seen_labels.add(lab)
        chosen_idx.append(int(j))
        chosen_scores.append(s)
        if len(chosen_idx) >= k:
            break

    return chosen_idx, chosen_scores

@torch.no_grad()
def _compute_label_hit_ratio_at_k(
    vt,
    token_model,
    images_t: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_label_sets: List[set],
    text_label_ids: List[int],
    k: int = 5,
) -> Tuple[float, int, int]:
    device = next(vt.parameters()).device
    use_amp = (device.type == "cuda")
    amp_dtype = torch.float32

    images_t = images_t.to(device, non_blocking=True)
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp)
    with ctx:
        feats, token_num = token_model.forward_test(images_t)
        out = vt(feats, input_ids, attention_mask, targets=None)

        candidate_keys = ["logits", "logits_per_image", "sims", "similarity", "scores"]
        if isinstance(out, dict):
            scores = None
            for k_name in candidate_keys:
                if k_name in out:
                    scores = out[k_name]
                    break
            if scores is None:
                raise RuntimeError(
                    f"VisionTextSigLIP forward 결과에서 score 행렬 키({candidate_keys})를 찾지 못했다."
                )
        else:
            scores = out

    if scores.dim() != 2:
        raise ValueError(f"scores 텐서는 [B, M] 2D 여야 하는데, shape={scores.shape} 입니다.")

    B, M = scores.shape
    if B == 0 or M == 0:
        return float('nan'), 0, 0

    if len(text_label_ids) != M:
        raise ValueError(
            f"text_label_ids 길이({len(text_label_ids)})와 scores의 두번째 차원 M({M})이 일치해야 한다."
        )

    scores_np = scores.detach().cpu().numpy()

    total_hits = 0
    total_gt = 0

    for i in range(B):
        gt = image_label_sets[i]
        if len(gt) == 0:
            continue

        s_i = scores_np[i]
        chosen_idx, _chosen_scores = _topk_unique_by_label(
            scores_1d=s_i,
            labels_of_texts=text_label_ids,
            k=k,
            min_score=None,
        )
        pred_labels = {int(text_label_ids[j]) for j in chosen_idx}
        hits = len(pred_labels.intersection(gt))
        total_hits += hits
        total_gt += len(gt)

    ratio = (total_hits / total_gt * 100.0) if total_gt > 0 else float("nan")
    return ratio, total_hits, total_gt

# ---------------------
# 이미지 Dataset (query/ref infer용)
# ---------------------

class ImageFromList(torch.utils.data.Dataset):
    def __init__(self, Image_paths=None, transforms=None, imsize=None, bbox=None, loader=None):
        super(ImageFromList, self).__init__()
        self.Image_paths = Image_paths or []
        self.transforms = transforms
        self.bbox = bbox
        self.imsize = imsize
        self.loader = loader if loader is not None else self.pil_loader
        self.len = len(self.Image_paths)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        path = self.Image_paths[index]
        img = self.loader(path)

        if self.bbox is not None:
            img = img.crop(self.bbox[index])

        if self.imsize is not None:
            if isinstance(self.imsize, int):
                imsize = (self.imsize, self.imsize)
            else:
                imsize = self.imsize
            img = T.Resize(imsize, interpolation=InterpolationMode.BICUBIC)(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return self.len

# ---------------------
# 코사인 거리/유사도
# ---------------------

@torch.no_grad()
def cosine_distance_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a, b: [N, D], [M, D]
    반환: [N, M] = 1 - cosine_similarity(a, b)
    (대규모를 위해 배치 단위로 계산)
    """
    assert a.dim() == 2 and b.dim() == 2
    device = a.device
    batch_size = 50

    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)

    sim = torch.empty((a.size(0), b.size(0)), device=device, dtype=a.dtype)

    for i in range(0, a.size(0), batch_size):
        end_i = min(i + batch_size, a.size(0))
        a_chunk = a[i:end_i]  # [Bi, D]
        for j in range(0, b.size(0), batch_size):
            end_j = min(j + batch_size, b.size(0))
            b_chunk = b[j:end_j]  # [Bj, D]
            dot_product = a_chunk @ b_chunk.t()  # [Bi, Bj]
            sim[i:end_i, j:end_j] = 1.0 - dot_product  # distance = 1 - cos

    return sim

def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = torch.nn.functional.normalize(A, dim=1)
    B = torch.nn.functional.normalize(B, dim=1)
    return A @ B.t()

# ---------------------
# fused vector 추출
# ---------------------

# @torch.no_grad()
# def extract_fused_vectors(
#     vt: VisionTextSigLIP,
#     token_model: Token,
#     loader: DataLoader,
#     t_all: torch.Tensor,     # (M, D) precomputed text embeddings
#     weight_img: float,
#     weight_txt: float,
#     attn_temp: float,
#     device: torch.device,
#     topk_attn: int = 0,      # >0 이면 attn 상위 K만 사용
# ) -> torch.Tensor:
#     """
#     학습 때 사용한 CrossAttention(qkv) 로 score를 만들고,
#     softmax( attn_temp * scores ) 로 attn을 만든 뒤,
#     필요시 상위 K만 남겨 재정규화하여 텍스트 가중합 벡터 h_attn을 만든다.

#       v       = vt._project_image_feats(Token.forward_test(img))   # (B, D)
#       scores  = vt._cross_attend_image_to_text(v, t_all)           # (B, M)
#       attn    = softmax(attn_temp * scores, -1)                    # (B, M)
#       attn(K) = top-K mask 후 row-wise normalize                   # (B, M)
#       h_attn  = attn(K) @ t_all                                    # (B, D)
#       fused   = norm( (1-λ)*v + λ*h_attn )
#     """
#     vt.eval()
#     token_model.eval()

#     use_amp = (device.type == "cuda")
#     try:
#         amp_dtype = torch.float32 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
#     except Exception:
#         amp_dtype = torch.float32

#     num_images = len(loader.dataset)
#     D = t_all.size(1)
#     all_fused = torch.zeros(num_images, D, dtype=torch.float32, device=device)

#     idx_start = 0
#     M = t_all.size(0)

#     for images in tqdm(loader, desc="Extract fused (xattn-based)", total=len(loader)):
#         images = images.to(device, non_blocking=True)
#         bsz = images.size(0)

#         with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
#             # 1) Token → image_feats
#             image_feats = token_model.forward_test(images)          # (B, Dv)

#             # 2) vt projection → v
#             v = vt._project_image_feats(image_feats)                # (B, D)

#             # 3) 학습된 qkv CrossAttention으로 raw scores 만들기
#             scores = vt._cross_attend_image_to_text(v, t_all)       # (B, M)

#         # 4) softmax + temperature → attn (B, M)
#         attn = torch.softmax(attn_temp * scores, dim=-1)            # (B, M)


#         # 🔹 5) topK attention (선택)
#         if (topk_attn is not None) and (topk_attn > 0) and (topk_attn < M):
#             topk_vals, topk_idx = torch.topk(attn, k=topk_attn, dim=-1)   # (B, K)

#             mask = torch.zeros_like(attn)
#             mask.scatter_(dim=-1, index=topk_idx, value=1.0)        # topK 위치만 1

#             attn = attn * mask
#             attn_sum = attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
#             attn = attn / attn_sum

#         # 6) 텍스트 가중합 벡터 h_attn
#         h_attn = attn @ t_all                                      # (B, D)
#         h_attn = F.normalize(h_attn, p=2, dim=1)

#         image_feats_norm = F.normalize(image_feats, p=2, dim=1)    # (B, Dv) = (B, D) 가정

#         # 7) 최종 fused 벡터
#         fused = (weight_img * image_feats_norm) + (weight_txt * h_attn)
#         fused = F.normalize(fused, p=2, dim=1)

#         all_fused[idx_start:idx_start + bsz] = fused
#         idx_start += bsz

#     return all_fused

@torch.no_grad()
def extract_fused_vectors(
    vt: VisionTextSigLIP,
    token_model: Token,
    loader: DataLoader,
    t_all: torch.Tensor,     # (M, D)
    weight_img: float,
    weight_txt: float,
    attn_temp: float,
    device: torch.device,
    topk_attn: int = 0,
) -> torch.Tensor:
    """
    변경된 버전:
      - attn @ t_all (가중합) 사용 X
      - attn이 가장 높은 text prototype TOP-5 선택
      - TOP-5 prototype 임베딩을 단순 평균(mean)
      - fused = norm((1-λ)*image_feats + λ*text_mean)
    """

    vt.eval()
    token_model.eval()

    use_amp = (device.type == "cuda")
    try:
        amp_dtype = torch.float32 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    except Exception:
        amp_dtype = torch.float32

    num_images = len(loader.dataset)
    D = t_all.size(1)
    all_fused = torch.zeros(num_images, D, dtype=torch.float32, device=device)

    idx_start = 0
    M = t_all.size(0)
    TOP_N = 3   # ★ 사용자 요청: TOP-5 단순 평균

    for images in tqdm(loader, desc="Extract fused (mean top5 prototypes)", total=len(loader)):
        images = images.to(device, non_blocking=True)
        bsz = images.size(0)

        # -----------------------------
        # 1) 이미지 피쳐
        # -----------------------------
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            image_feats, token_num = token_model.forward_test(images)      # (B, Dv)
            v = vt._project_image_feats(image_feats)            # (B, D)

            scores = vt._cross_attend_image_to_text(v, t_all)   # (B, M)

        # -----------------------------
        # 2) softmax + temperature
        # -----------------------------
        attn = torch.softmax(attn_temp * scores, dim=-1)        # (B, M)

        # -----------------------------
        # 3) attn이 높은 TOP-5 index 선택
        # -----------------------------
        top_vals, top_idx = torch.topk(attn, k=TOP_N, dim=-1)   # (B,5), (B,5)

        # -----------------------------
        # 4) TOP-5 prototype embedding 가져오기 → mean
        # -----------------------------
        # t_all: (M, D)
        # top_idx: (B, 5)
        top5_embs = t_all[top_idx]           # (B,5,D)
        text_mean = top5_embs.mean(dim=1)    # (B, D)
        text_mean = F.normalize(text_mean, p=2, dim=1)

        # -----------------------------
        # 5) 이미지 벡터 normalize
        # -----------------------------
        image_feats_norm = F.normalize(image_feats, p=2, dim=1)

        # -----------------------------
        # 6) 최종 fused
        # -----------------------------
        fused = (weight_img * image_feats_norm) + (weight_txt * text_mean)
        fused = F.normalize(fused, p=2, dim=1)

        all_fused[idx_start:idx_start + bsz] = fused
        idx_start += bsz

    return all_fused





# ---------------------
# 모델 빌드 함수
# ---------------------

def build_model(
    device: str,
    vt_ckpt_path: Optional[str] = None,
    token_ckpt_path: Optional[str] = None,
    strict_load: bool = False
):
    token_model = Token(outputdim=1024, classifier_num=81313, mode='train').to(device)

    for p in token_model.parameters():
        p.requires_grad = False
    token_model.eval()

    if token_ckpt_path:
        try:
            load_token_from_path(token_model, token_ckpt_path, map_location="cpu", strict=strict_load)
        except Exception as e:
            print(f"[Token] Failed to load TOKEN from '{token_ckpt_path}': {e}")

    text_encoder = LLMTextEncoder(
        model_name=MODEL_NAME,
        device=device,
        dtype=torch.bfloat16,
        train_llm=True,
        use_lora=True,
        lora_r=8, lora_alpha=16, lora_dropout=0.1,
        pooling="mean",
    )
    vt = VisionTextSigLIP(
        text_encoder=text_encoder,
        vision_dim=1024,
        proj_out_dim=1024,
        temperature_init=0.06
    ).to(device).train()

    if vt_ckpt_path:
        try:
            load_vt_from_path(vt, vt_ckpt_path, map_location="cpu", strict=strict_load)
        except Exception as e:
            print(f"[VT] Failed to load VT from '{vt_ckpt_path}': {e}")

    return vt, token_model

# ---------------------
# 학습 루프
# ---------------------

def main_train(
    image_size=image_size,
    norm_mean=tuple(NORM_MEAN),
    norm_std=tuple(NORM_STD),
    white_bg_fill=True,
    allow_flip=False,
    save_interval=20,
    resume_tag: Optional[str] = None,
    resume_path: Optional[str] = None,
    resume_all: bool = False,
    vt_ckpt_path: Optional[str] = None,
    token_ckpt_path: Optional[str] = None,
    strict_load: bool = False,
    save_full_state: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = (device == "cuda")

    vt, token_model = build_model(
        device=device,
        vt_ckpt_path=vt_ckpt_path,
        token_ckpt_path=token_ckpt_path,
        strict_load=strict_load
    )

    assert os.path.isfile(ITEMS_IMG_PATH), f"not found: {ITEMS_IMG_PATH}"
    assert os.path.isfile(ITEMS_TXT_PATH), f"not found: {ITEMS_TXT_PATH}"
    items_img = load_jsonl(ITEMS_IMG_PATH)
    items_txt = load_jsonl(ITEMS_TXT_PATH)
    label2texts = build_label2texts(items_txt)

    tfm = T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.RandomApply([
            T.RandomAffine(
                degrees=2,
                translate=(0.01, 0.01),
                scale=(0.98, 1.02),
                shear=1,
                interpolation=InterpolationMode.BICUBIC,
                fill=255 if white_bg_fill else 0
            )
        ], p=0.30),
        T.RandomApply([
            T.RandomPerspective(distortion_scale=0.02, p=1.0)
        ], p=0.10),
        T.RandomApply([
            T.ColorJitter(brightness=0.05, contrast=0.05)
        ], p=0.30),
        T.RandomHorizontalFlip(p=0.15 if allow_flip else 0.0),
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.15),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    img_ds = ImageDatasetMultiLabel(items_img, image_transform=tfm)
    img_loader = DataLoader(
        img_ds, batch_size=BATCH_SIZE_IMG, shuffle=True, num_workers=NUM_WORKERS_IMG,
        collate_fn=ImageCollatorMulti(), pin_memory=(device == "cuda"),
        persistent_workers=(NUM_WORKERS_IMG > 0)
    )

    steps_per_epoch = len(img_loader)
    num_images = len(img_ds)
    print(f"[Data] #images={num_images} | batch_size={BATCH_SIZE_IMG} | steps_per_epoch={steps_per_epoch}")

    trainable = [p for p in vt.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch, global_step = _load_resume_checkpoint(
        vt=vt,
        token_model=token_model,
        optimizer=optimizer,
        scaler=scaler,
        resume_path=resume_path if resume_path else None,
        resume_tag=resume_tag if resume_tag else None,
        device=device,
        resume_all=resume_all,
    )

    total_steps = EPOCHS * steps_per_epoch
    ema_iter_time = None
    start_time = time.perf_counter()
    tokenizer = vt.text_encoder.tokenizer

    grad_probe = GradProbe(token_model, name_prefix="token")
    if DEBUG_BACKPROP:
        grad_probe.attach()
    snap = ParamSnapshot(token_model, max_params=DEBUG_SAMPLE_PARAMS)

    for epoch in range(start_epoch, EPOCHS + 1):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()

        cum_hits = 0
        cum_gt   = 0

        for step, batch_img in enumerate(img_loader, start=1):
            t0 = time.perf_counter()

            batch_txt = _build_text_batch_one_per_label(
                batch_img=batch_img,
                label2texts=label2texts,
                tokenizer=tokenizer,
                max_length=64
            )
            if batch_txt is None:
                continue

            if DEBUG_BACKPROP and (global_step % DEBUG_EVERY_STEPS == 0):
                grad_probe.reset()
                snap.take_before()

            logs = train_step_linked(vt, token_model, batch_img, batch_txt, optimizer, scaler)

            cur_loss = float(logs.get("loss", float("nan")))
            cur_temp = float(logs.get("temp", float("nan")))

            if DEBUG_BACKPROP and (global_step % DEBUG_EVERY_STEPS == 0):
                snap.take_after()
                _ = snap.changed_flags()
                _ = grad_probe.summary()

            text_label_ids = getattr(batch_txt, "label_ids", None)
            batch_ratio = float("nan")
            if text_label_ids is not None:
                img_label_sets = _extract_image_label_sets(batch_img)

                if hasattr(batch_img, "images"):
                    images_t = batch_img.images
                elif isinstance(batch_img, dict) and "images" in batch_img:
                    images_t = batch_img["images"]
                else:
                    raise TypeError("batch_img에서 images 텐서를 찾을 수 없다.")

                device_t = next(vt.parameters()).device
                images_t = images_t.to(device_t, non_blocking=True)
                input_ids = batch_txt.input_ids.to(device_t, non_blocking=True)
                attention_mask = batch_txt.attention_mask.to(device_t, non_blocking=True)

                ratio, hits, gt = _compute_label_hit_ratio_at_k(
                    vt=vt,
                    token_model=token_model,
                    images_t=images_t,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_label_sets=img_label_sets,
                    text_label_ids=text_label_ids,
                    k=LABEL_HIT_AT_K,
                )
                batch_ratio = ratio
                cum_hits += hits
                cum_gt   += gt

            global_step += 1
            iter_time = time.perf_counter() - t0
            ema_iter_time = iter_time if ema_iter_time is None else (0.9 * ema_iter_time + 0.1 * iter_time)

            steps_done = (epoch - 1) * steps_per_epoch + step
            steps_left = max(0, total_steps - steps_done)
            eta_sec = steps_left * (ema_iter_time if ema_iter_time is not None else iter_time)

            max_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
            lr_cur = optimizer.param_groups[0]["lr"]

            if (step % LOG_EVERY_STEPS) == 0:
                cum_ratio = (cum_hits / cum_gt * 100.0) if cum_gt > 0 else float("nan")
                print(
                    f">> Train Epoch: [{epoch}] "
                    f"[{step}/{steps_per_epoch}] "
                    f"eta: {format_eta(eta_sec)} "
                    f"VT contrastive loss: {cur_loss:.4f} "
                    f"Label-Hit@{LABEL_HIT_AT_K}: batch {batch_ratio:6.3f}% | epoch {cum_ratio:6.3f}% "
                    f"iter time: {iter_time:.4f} s "
                    f"lr: {lr_cur:.2e} "
                    f"max mem: {int(max_mem_mb)} MB"
                )

        epoch_time = time.perf_counter() - epoch_start
        print(f">> Epoch [{epoch}] done in {format_eta(epoch_time)}")

        if epoch % save_interval == 0:
            save_weights(vt, token_model, optimizer, scaler, epoch, global_step, tag=f"epoch{epoch:03d}", save_full_state=save_full_state)

    total_time = time.perf_counter() - start_time
    print(f">> Training done in {format_eta(total_time)}")
    save_weights(vt, token_model, optimizer, scaler, epoch=EPOCHS, global_step=global_step, tag="last", save_full_state=save_full_state)

# ---------------------
# 텍스트 프로토타입 임베딩
# ---------------------

@torch.no_grad()
def precompute_text_embeddings_torch(
    vt: VisionTextSigLIP,
    items_txt,
    device: torch.device,
    text_batch_size: int = 256,
    max_length: int = 128,
):
    tokenizer = vt.text_encoder.tokenizer
    collate_txt = TextCollatorSingle(tokenizer, max_length=max_length)

    label_to_texts = defaultdict(list)
    for it in items_txt:
        lab = int(it["label"])
        txt = it["text"]
        label_to_texts[lab].append(txt)

    selected_items = []
    for lab, txt_list in label_to_texts.items():
        chosen_text = random.choice(txt_list)
        selected_items.append({"text": chosen_text, "label": lab})

    proto_labels = [it["label"] for it in selected_items]
    proto_texts  = [it["text"]  for it in selected_items]

    use_amp = (device.type == "cuda")
    try:
        amp_dtype = torch.float32 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    except Exception:
        amp_dtype = torch.float16

    all_embs = []
    total = len(selected_items)
    s = 0

    while s < total:
        e = min(s + text_batch_size, total)
        batch = selected_items[s:e]

        batch_txt = collate_txt(batch)
        input_ids = batch_txt.input_ids.to(device, non_blocking=True)
        attention_mask = batch_txt.attention_mask.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            t_batch = vt.encode_texts(input_ids, attention_mask)

        all_embs.append(t_batch.float())
        s = e

    t_all = torch.cat(all_embs, dim=0)

    print(f"[Text] unique labels = {len(selected_items)}")
    print(f"[Text] precomputed text embeddings: shape={t_all.shape}")
    return t_all, proto_labels, proto_texts

# ---------------------
# Text Self-Attention 적용 유틸
# ---------------------

@torch.no_grad()
def apply_text_self_attention(vt: VisionTextSigLIP, t_all: torch.Tensor) -> torch.Tensor:
    """
    학습 시 forward()에서 사용하는 것과 동일하게,
    텍스트 프로토타입 전체(M, D)에 대해 self-attention을 한 번 적용한다.
    """
    if getattr(vt, "use_text_self_attn", False) and (vt.text_self_attn_block is not None):
        # TextSelfAttentionBlock은 (M, D) 또는 (B, M, D)를 받도록 되어 있음
        t_all = vt.text_self_attn_block(t_all)   # (M, D) -> (M, D)
    return t_all

# ---------------------
# Infer: fused 벡터 (단일 이미지 세트용)
# ---------------------

@torch.no_grad()
def run_infer(
    ckpt_tag: str = "last",
    topk: int = 5,
    text_batch: int = 256,
    image_paths: Optional[List[str]] = None,
    max_demo_images: int = 8,
    vt_ckpt_path: Optional[str] = None,
    token_ckpt_path: Optional[str] = None,
    strict_load: bool = False,
    fused_outdir: Optional[str] = None,
    lam: float = 0.2,
    attn_temp: float = 50.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vt, token_model = build_model(
        device=device,
        vt_ckpt_path=vt_ckpt_path,
        token_ckpt_path=token_ckpt_path,
        strict_load=strict_load,
    )
    if not vt_ckpt_path and not token_ckpt_path:
        load_train_state_or_pair(vt, token_model, tag=ckpt_tag)

    vt.eval()
    token_model.eval()

    assert os.path.isfile(ITEMS_TXT_PATH), f"not found: {ITEMS_TXT_PATH}"
    items_txt = load_jsonl(ITEMS_TXT_PATH)

    t_all, proto_labels, proto_texts = precompute_text_embeddings_torch(
        vt=vt,
        items_txt=items_txt,
        device=torch.device(device),
        text_batch_size=text_batch,
        max_length=128,
    )
    M, D = t_all.shape
    print(f"[Infer] text prototypes: M={M}, D={D}")

    t_all = t_all.to(device, non_blocking=True)
    t_all = apply_text_self_attention(vt, t_all)   # (M, D) -> (M, D)

    if not image_paths:
        assert os.path.isfile(ITEMS_IMG_PATH), f"not found: {ITEMS_IMG_PATH}"
        items_img = load_jsonl(ITEMS_IMG_PATH)
        image_paths = [it["image_path"] for it in items_img][:max_demo_images]

    print(f"[Infer] #images: {len(image_paths)}")

    tfm = T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    image_tensors, valid_paths = [], []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                image_tensors.append(tfm(im))
                valid_paths.append(p)
        except Exception as e:
            print(f"[Warn] 이미지 열기 실패: {p} ({e})")

    if len(image_tensors) == 0:
        print("[Infer] 사용할 이미지가 없습니다.")
        return

    images = torch.stack(image_tensors, dim=0).to(device, non_blocking=True)
    B = images.size(0)
    print(f"[Infer] images tensor shape = {images.shape}")

    use_amp = (device == "cuda")
    try:
        amp_dtype = torch.float32 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    except Exception:
        amp_dtype = torch.float32

    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        image_feats, token_num = token_model.forward_test(images)
        v = vt._project_image_feats(image_feats)
        scores = vt._cross_attend_image_to_text(v, t_all)
        print("[Infer] scores shape:", scores.shape)

    attn = torch.softmax(attn_temp * scores, dim=-1)
    print("attn max:", attn.max().item(), "attn min:", attn.min().item())

    h_attn = attn @ t_all
    h_attn = F.normalize(h_attn, p=2, dim=1)

    weight_img = 1.0 - lam
    weight_txt = lam
    fused = (weight_img * image_feats) + (weight_txt * h_attn)
    fused = F.normalize(fused, p=2, dim=1)

    fused_vecs = fused.detach().cpu().to(torch.float32).numpy()

    if fused_outdir is None or len(str(fused_outdir).strip()) == 0:
        fused_outdir = os.path.join(os.path.dirname(valid_paths[0]), "fused_vecs")
    os.makedirs(fused_outdir, exist_ok=True)

    npy_path = os.path.join(fused_outdir, "all_images_fused_xattn.npy")
    np.save(npy_path, fused_vecs)
    print(f"[Infer] saved fused image embeddings: {npy_path} (shape={fused_vecs.shape})")

    meta_all = {}
    attn_np = attn.detach().cpu().numpy()   # (B, M)

    for i, img_path in enumerate(valid_paths):
        attn_i = attn_np[i]  # shape (M,)

        chosen_idx, _ = _topk_unique_by_label(
            scores_1d=attn_i,
            labels_of_texts=proto_labels,
            k=topk,
            min_score=None,
        )

        meta_all[os.path.basename(img_path)] = [
            {
                "rank": r + 1,
                "text_index": int(ti),
                "label": int(proto_labels[ti]),
                "attn": float(attn_i[ti]),
                "text": proto_texts[ti],
            }
            for r, ti in enumerate(chosen_idx)
        ]

    meta_path = os.path.join(fused_outdir, f"all_images_fused_xattn.meta_top{topk}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    print(f"[Infer] saved meta json (Top-{topk} prototypes per image): {meta_path}")
    print(f"[Infer] lambda={lam:.3f} → fused = norm((1-λ)*v + λ*h_attn), attn_temp={attn_temp:.1f}")

# ---------------------
# 매칭/기타 유틸
# ---------------------

def _extract_name_by_regex(p: str) -> str:
    m = re.search(r"/([^/]+)\.", p)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(p))[0]

def _load_image_paths_from_jsonl(path: str) -> List[str]:
    assert os.path.isfile(path), f"jsonl not found: {path}"
    rows = load_jsonl(path)
    out = []
    for r in rows:
        p = r.get("image_path", None)
        if isinstance(p, str) and len(p) > 0:
            out.append(p)
    if len(out) == 0:
        raise RuntimeError(f"no image_path in {path}")
    return out

# ---------------------
# query/ref 매칭 (fused + log_print)
# ---------------------

@torch.no_grad()
def run_match_fused(args):
    """
    - ITEMS_TXT_PATH 에서 텍스트 프로토타입 임베딩 선계산
    - query_jsonl / ref_jsonl 에서 이미지 경로 로드
    - cross-attention 기반 fused 벡터 추출
    - cosine distance 로 query/ref 매칭 후 log_print 호출
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 모델 로드
    vt, token_model = build_model(
        device=device,
        vt_ckpt_path=args.vt_ckpt_path if len(args.vt_ckpt_path) > 0 else None,
        token_ckpt_path=args.token_ckpt_path if len(args.token_ckpt_path) > 0 else None,
        strict_load=bool(args.strict_load),
    )
    if (len(args.vt_ckpt_path) == 0) and (len(args.token_ckpt_path) == 0):
        load_train_state_or_pair(vt, token_model, tag=args.ckpt_tag)

    vt.eval()
    token_model.eval()

    # 2) 텍스트 프로토타입 임베딩
    assert os.path.isfile(ITEMS_TXT_PATH), f"not found: {ITEMS_TXT_PATH}"
    items_txt = load_jsonl(ITEMS_TXT_PATH)

    t_all, proto_labels, proto_texts = precompute_text_embeddings_torch(
        vt=vt,
        items_txt=items_txt,
        device=torch.device(device),
        text_batch_size=args.text_batch,
        max_length=128,
    )
    M, D = t_all.shape
    print(f"[Match] text prototypes: M={M}, D={D}")

    t_all = t_all.to(device, non_blocking=True)
    t_all = apply_text_self_attention(vt, t_all)   # (M, D) -> (M, D)

    # 3) query/ref 이미지 경로 로드 (JSONL)
    assert len(args.query_jsonl) > 0 and len(args.ref_jsonl) > 0, \
        "match 모드에서는 --query_jsonl, --ref_jsonl 을 지정해야 한다."

    query_paths = _load_image_paths_from_jsonl(args.query_jsonl)
    ref_paths   = _load_image_paths_from_jsonl(args.ref_jsonl)

    print(f"[Match] #query={len(query_paths)} , #ref={len(ref_paths)}")

    if len(query_paths) == 0 or len(ref_paths) == 0:
        raise RuntimeError(
            f"[Match] query/ref 이미지가 비어 있습니다.\n"
            f"  query_jsonl={args.query_jsonl} (found {len(query_paths)} files)\n"
            f"  ref_jsonl={args.ref_jsonl} (found {len(ref_paths)} files)"
        )

    # 이름 추출 (log_print 용)
    query_order = [_extract_name_by_regex(p) for p in query_paths]
    ref_names   = [_extract_name_by_regex(p) for p in ref_paths]
    ref_indices = {name: i for i, name in enumerate(ref_names)}

    # 4) 이미지 transform & dataloader
    tfm = T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    bs = BATCH_SIZE_IMG

    query_loader = DataLoader(
        ImageFromList(Image_paths=query_paths, imsize=image_size, bbox=None, transforms=tfm),
        batch_size=bs, shuffle=False, num_workers=NUM_WORKERS_IMG,
        pin_memory=(device == "cuda")
    )
    ref_loader = DataLoader(
        ImageFromList(Image_paths=ref_paths, imsize=image_size, bbox=None, transforms=tfm),
        batch_size=bs, shuffle=False, num_workers=NUM_WORKERS_IMG,
        pin_memory=(device == "cuda")
    )

    # 5) fused 벡터 추출
    lam = args.lam
    weight_img = 1.0 - lam
    weight_txt = lam
    print(f"[Match] lambda={lam:.3f} → fused = norm((1-λ)*v + λ*h_attn)")
    print(f"        weight_img={weight_img:.3f}, weight_txt={weight_txt:.3f}")
    print(f"        attn_temp={args.attn_temp:.2f}, topk_attn={args.topk_attn}")

    query_vecs = extract_fused_vectors(
        vt=vt,
        token_model=token_model,
        loader=query_loader,
        t_all=t_all,
        weight_img=weight_img,
        weight_txt=weight_txt,
        attn_temp=args.attn_temp,
        device=torch.device(device),
        topk_attn=args.topk_attn if args.topk_attn > 0 else 0
    )
    ref_vecs = extract_fused_vectors(
        vt=vt,
        token_model=token_model,
        loader=ref_loader,
        t_all=t_all,
        weight_img=weight_img,
        weight_txt=weight_txt,
        attn_temp=args.attn_temp,
        device=torch.device(device),
        topk_attn=args.topk_attn if args.topk_attn > 0 else 0
    )

    # 6) 거리 계산 & 정렬
    distances = cosine_distance_matrix(query_vecs, ref_vecs)   # [Q, R], 1 - cos
    sorted_distances, sorted_indices = torch.sort(distances, dim=1)  # 오름차순: 가까운 것부터

    # 7) log_print 호출
    log_print(sorted_indices, sorted_distances, query_order, ref_indices, ref_names, args)
    print("[Match] log_print 완료")

# ===================== #
#    엔트리 포인트       #
# ===================== #

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="train",
                    choices=["train", "infer", "both", "match"],
                    help="train: 학습, infer: 데모 추론, both: 학습 후 데모 추론, match: query/ref JSONL 매칭")
    ap.add_argument("--ckpt_tag", type=str, default="last", help="(infer/match) 태그로 vt_/token_ 또는 train_state_ 로드")
    ap.add_argument("--topk", type=int, default=5, help="각 이미지별 상위 출력 라벨 개수(라벨 중복 없음)")
    ap.add_argument("--text_batch", type=int, default=128, help="추론 시 텍스트 배치 크기")
    ap.add_argument("--infer_images", type=str, default="/root/project/llm_prompt/test_crime",
                    help="추론용 이미지 경로(쉼표/공백 구분) 또는 글롭 패턴. 미지정 시 items_img.jsonl 일부로 데모")
    ap.add_argument("--max_demo_images", type=int, default=1000, help="infer_images 미지정 시 데모에 사용할 이미지 수")

    ap.add_argument("--query_jsonl", type=str, default="/root/project/llm_prompt_new/json_file/image_paths.jsonl",
                    help="(match) query 이미지 목록 JSONL 경로")
    ap.add_argument("--ref_jsonl",   type=str, default="/root/project/llm_prompt/json_file/items_img_all_rgb_group.jsonl",
                    help="(match) ref   이미지 목록 JSONL 경로")
    ap.add_argument("--desc_sim", type=int, default=1, help="1이면 유사도 내림차순 정렬")
    ap.add_argument("--log_topk", type=int, default=5, help="log_print에서 표시할 Top-K")

    ap.add_argument("--save", type=str,
                    default="/home/miruware/ieoo0321/Origin_Token/Output",
                    help="The path to save vector output (used in log_print)")
    ap.add_argument("--testcsv", type=str,
                    default="/root/project/llm_prompt/label_test_multimodal.csv",
                    help="gt csv path (used in log_print)")

    ap.add_argument("--resume_path", type=str,
                    default="",
                    help="직접 지정한 train_state_*.pt 경로")
    ap.add_argument("--resume_tag", type=str, default="",
                    help="CKPT_DIR/train_state_{TAG}.pt 로드")
    ap.add_argument("--resume_all", type=int, default=0,
                    help="1이면 optimizer/scaler까지 함께 로드")

    ap.add_argument("--vt_ckpt_path", type=str,
                    default="/root/project/llm_prompt/llm_machine/checkpoint_siglip/vt_epoch020_final_self_attention_multimodal_1.pth",
                    help="vt_*.pth 등 개별 가중치 파일 경로")
    ap.add_argument("--token_ckpt_path", type=str,
                    default="/root/project/llm_prompt/llm_machine/checkpoint_siglip/token_epoch020_final_self_attention_multimodal_1.pth",
                    help="token_*.pth 등 개별 가중치 파일 경로")
    ap.add_argument("--strict_load", type=int, default=0, help="1이면 state_dict strict 로드")
    ap.add_argument("--save_full_state", type=int, default=1,
                    help="1이면 train_state_*.pt도 함께 저장, 0이면 vt/token만 저장")

    ap.add_argument("--lam", type=float, default=0
    ,
                    help="fused = norm((1-lam)*v + lam*h_attn) 에서 λ")
    ap.add_argument("--attn_temp", type=float, default=30.0,
                    help="cosine-attention softmax temperature (크면 더 샤프)")
    ap.add_argument("--topk_attn", type=int, default=4,
                    help=">0 이면 attn 상위 K만 사용하여 h_attn 계산")

    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()

    if args.mode in ("train", "both"):
        resume_path = args.resume_path if len(args.resume_path) > 0 else None
        resume_tag = args.resume_tag if len(args.resume_tag) > 0 else None

        main_train(
            resume_path=resume_path,
            resume_tag=resume_tag,
            resume_all=bool(args.resume_all),
            vt_ckpt_path=args.vt_ckpt_path if len(args.vt_ckpt_path) > 0 else None,
            token_ckpt_path=args.token_ckpt_path if len(args.token_ckpt_path) > 0 else None,
            strict_load=bool(args.strict_load),
            save_full_state=bool(args.save_full_state),
        )

    if args.mode in ("infer", "both"):
        paths = _gather_paths(args.infer_images)
        run_infer(
            ckpt_tag=args.ckpt_tag,
            topk=args.topk,
            text_batch=args.text_batch,
            image_paths=paths if len(paths) > 0 else None,
            max_demo_images=args.max_demo_images,
            vt_ckpt_path=args.vt_ckpt_path if len(args.vt_ckpt_path) > 0 else None,
            token_ckpt_path=args.token_ckpt_path if len(args.token_ckpt_path) > 0 else None,
            strict_load=bool(args.strict_load),
            fused_outdir=None,
            lam=args.lam,
            attn_temp=args.attn_temp,
        )

    if args.mode == "match":
        run_match_fused(args)


