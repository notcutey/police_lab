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

from llm_machine import LLMTextEncoder
from llm_machine import VisionTextSigLIP
from llm_machine.data_linked import ImageDatasetMultiLabel, ImageCollatorMulti, TextCollatorSingle
from networks import Token
from llm_machine import train_step_linked
from llm_machine import log_print

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
image_size = (1024, 512) 
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD  = [0.26862954, 0.26130258, 0.27577711]

ITEMS_IMG_PATH = "/root/project/llm_prompt_new/json_file/label_test_multimodal.jsonl"
ITEMS_TXT_PATH = "/root/project/llm_prompt_new/json_file/items_txt_rgb+crime_final.jsonl"

TOKEN_CKPT_PATH = None  # (구) 전역 자동 로드 제거, 필요 시 인자 사용

# 결과 저장 디렉터리
CKPT_DIR = "/root/project/llm_prompt/llm_machine/checkpoint_siglip"
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS = 600
BATCH_SIZE_IMG = 128
NUM_WORKERS_IMG = 4

LR = 1e-4
WEIGHT_DECAY = 0.01

# ====== 메트릭/로그 설정 ======
LABEL_HIT_AT_K = 5          # 상위 K개 예측에서 맞춘 정답 라벨 비율(R@K)
LOG_EVERY_STEPS = 10        # 10 스텝 단위 로그

# 디버깅 토글
DEBUG_BACKPROP = True
DEBUG_SAMPLE_PARAMS = 3
DEBUG_EVERY_STEPS = 1

# ===================== #
#     유틸 함수 묶음     #
# ===== [ADD] progress/logging utils =====
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
# ===================== #







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
    """vt/token 개별 저장 + (옵션) 통합 상태 저장"""
    vt_sd = vt.state_dict()
    token_sd = (token_model.module if hasattr(token_model, "module") else token_model).state_dict()

    vt_path = os.path.join(CKPT_DIR, f"vt_{tag}.pt")
    token_path = os.path.join(CKPT_DIR, f"token_{tag}.pt")
    torch.save(vt_sd, vt_path)
    torch.save(token_sd, token_path)
    print(f"[Save] vt -> {vt_path}")
    print(f"[Save] token -> {token_path}")

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
    vt.load_state_dict(sd, strict=strict)
    print(f"[LoadVT] loaded vt weights from {path}")


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
        filtered = {k: v for k, v in sd.items()
                    if (k in model_dict) and (model_dict[k].shape == v.shape)
                    and not any(x in k for x in ["classifier", "fc", "head", "heads", "arcface"])}
    incompatible = model_wo_ddp.load_state_dict(filtered, strict=False if not strict else True)
    miss = list(getattr(incompatible, "missing_keys", []))
    unexp = list(getattr(incompatible, "unexpected_keys", []))
    print(f"[LoadToken] loaded {len(filtered)} keys from {path}")
    if miss:
        print(f"[LoadToken] Missing keys: {len(miss)} (first 10): {miss[:10]}")
    if unexp:
        print(f"[LoadToken] Unexpected keys: {len(unexp)} (first 10): {unexp[:10]}")


def load_train_state_or_pair(vt, token_model, tag: str = "last") -> Optional[int]:
    state_path = os.path.join(CKPT_DIR, f"train_state_{tag}.pt")
    vt_path    = os.path.join(CKPT_DIR, f"vt_{tag}.pt")
    token_path = os.path.join(CKPT_DIR, f"token_{tag}.pt")

    epoch = None
    if os.path.isfile(state_path):
        raw = _safe_torch_load(state_path, map_location="cpu")
        if "vt" in raw:
            vt.load_state_dict(raw["vt"], strict=False)
        if "token" in raw:
            (token_model.module if hasattr(token_model, "module") else token_model).load_state_dict(raw["token"], strict=False)
        epoch = int(raw.get("epoch", 0))
        print(f"[Load] full train_state from {state_path} (epoch={epoch})")
        return epoch

    ok = False
    if os.path.isfile(vt_path):
        vt.load_state_dict(_safe_torch_load(vt_path, map_location="cpu"), strict=False)
        ok = True
    if os.path.isfile(token_path):
        (token_model.module if hasattr(token_model, "module") else token_model).load_state_dict(
            _safe_torch_load(token_path, map_location="cpu"), strict=False
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
    """통합 체크포인트로 '정확히 이어서' 학습할 때만 사용한다."""
    epoch = 0
    global_step = 0
    if resume_path is not None and len(str(resume_path)) > 0:
        assert os.path.isfile(resume_path), f"resume_path not found: {resume_path}"
        print(f"[Resume] Loading checkpoint from file: {resume_path}")
        raw = _safe_torch_load(resume_path, map_location=device)

        if "vt" in raw:
            vt.load_state_dict(raw["vt"], strict=False)
        if "token" in raw:
            (token_model.module if hasattr(token_model, "module") else token_model).load_state_dict(raw["token"], strict=False)

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
                (token_model.module if hasattr(token_model, "module") else token_model).load_state_dict(raw["token"], strict=False)

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
        lab = int(it["label"])  # 텍스트 라벨
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
def _compute_label_hit_ratio_at_k(
    vt,
    images_t: torch.Tensor,           # [B, 3, H, W]
    input_ids: torch.Tensor,          # [M, L]
    attention_mask: torch.Tensor,     # [M, L]
    image_label_sets: List[set],      # 길이 B
    text_label_ids: List[int],        # 길이 M
    k: int = 5,
) -> Tuple[float, int, int]:
    device = next(vt.parameters()).device
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16
    ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp)

    with ctx:
        vt.eval()
        out = vt(images_t, input_ids, attention_mask, targets=None)
        vt.train()

    candidate_keys = ["logits", "logits_per_image", "sims", "similarity", "scores"]
    for kname in candidate_keys:
        if isinstance(out, dict) and (kname in out):
            scores = out[kname]
            break
    else:
        scores = out if torch.is_tensor(out) else None
    if scores is None:
        raise RuntimeError("VisionTextSigLIP forward 결과에서 점수 행렬 키를 찾지 못했다.")

    B, M = scores.shape
    kk = min(k, M)
    _, topk_idx = torch.topk(scores, k=kk, dim=1)  # [B, kk]

    text_label_ids_t = torch.tensor(text_label_ids, device=scores.device)
    topk_label_ids = text_label_ids_t[topk_idx]  # [B, kk]

    total_hits = 0
    total_gt = 0
    topk_label_ids = topk_label_ids.cpu().tolist()
    for i in range(B):
        gt = image_label_sets[i]
        if len(gt) == 0:
            continue
        preds_k = set(topk_label_ids[i])
        hits = len(preds_k.intersection(gt))
        total_hits += hits
        total_gt += len(gt)

    ratio = (total_hits / total_gt * 100.0) if total_gt > 0 else float("nan")
    return ratio, total_hits, total_gt

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

    # === [Change] token: 'backbone'으로 시작하는 가중치만 freeze ===
    for name, p in token_model.named_parameters():
        if name.startswith("backbone"):
            p.requires_grad = False   # 백본만 고정
        else:
            p.requires_grad = True    # 나머지는 학습

    token_model.train()  # 일부 모듈 학습을 위해 train 모드 유지

    # (옵션) 사전 가중치 로드
    if token_ckpt_path:
        try:
            load_token_from_path(token_model, token_ckpt_path, map_location="cpu", strict=strict_load)
        except Exception as e:
            print(f"[Token] Failed to load TOKEN from '{token_ckpt_path}': {e}")

    text_encoder = LLMTextEncoder(
        model_name=MODEL_NAME,
        device=device,
        dtype=torch.bfloat16,
        train_llm=True,   # LoRA on
        use_lora=True,
        lora_r=8, lora_alpha=16, lora_dropout=0.1,
        pooling="mean",
    )
    vt = VisionTextSigLIP(
        token_model=token_model,       # 내부에 등록되므로 optimizer에 함께 반영됨
        text_encoder=text_encoder,
        vision_dim=1024,
        proj_out_dim=1024,
        temperature_init=0.06,
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
    image_size=image_size,  # [Change] 기본값 1024 고정 사용
    norm_mean=tuple(NORM_MEAN),
    norm_std=tuple(NORM_STD),
    white_bg_fill=True,
    allow_flip=False,
    save_interval=4,
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

    # 1) 모델 구성 + 분리 로드
    vt, token_model = build_model(
        device=device,
        vt_ckpt_path=vt_ckpt_path,
        token_ckpt_path=token_ckpt_path,
        strict_load=strict_load
    )

    # 2) 데이터 로드
    assert os.path.isfile(ITEMS_IMG_PATH), f"not found: {ITEMS_IMG_PATH}"
    assert os.path.isfile(ITEMS_TXT_PATH), f"not found: {ITEMS_TXT_PATH}"
    items_img = load_jsonl(ITEMS_IMG_PATH)   # [{"image_path": str, "labels": [int,...]}, ...]
    items_txt = load_jsonl(ITEMS_TXT_PATH)   # [{"text": str, "label": int}, ...]
    label2texts = build_label2texts(items_txt)

    # --- Train-time transform: Resize + Normalize 만 수행 ---
    tfm = T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.RandomApply([
            T.RandomAffine(
                degrees=2,                 # ±2도 내 작은 회전
                translate=(0.01, 0.01),    # 최대 1% 평행이동
                scale=(0.98, 1.02),        # 2% 이내 스케일 변화
                shear=1,                   # 미세 기울기
                interpolation=InterpolationMode.BICUBIC,
                fill=255 if white_bg_fill else 0
            )
        ], p=0.30),

        T.RandomApply([
            T.RandomPerspective(distortion_scale=0.02, p=1.0)  # 아주 약한 원근 왜곡
        ], p=0.10),

        T.RandomApply([
            T.ColorJitter(brightness=0.05, contrast=0.05)      # 밝기/대비 미세 조정
        ], p=0.30),

        T.RandomHorizontalFlip(p=0.15 if allow_flip else 0.0), # 좌우 반전(옵션)

        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))    # 약한 블러
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

    # 3) 옵티마이저 & AMP
    trainable = [p for p in vt.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # 4) (선택) 통합 체크포인트로 이어서 학습
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

    # ===== 계측 도구 준비 =====
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

            logs = train_step_linked(vt, batch_img, batch_txt, optimizer, scaler)
            cur_loss = float(logs.get("arcface_loss", logs.get("loss", float("nan"))))
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
                    f"ArcFace loss: {cur_loss:.4f} "
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
# 추론 유틸
# ---------------------

@torch.no_grad()
def _forward_scores(vt, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    vt.eval()
    out = vt(images, input_ids, attention_mask, targets=None)
    candidate_keys = ["logits", "logits_per_image", "sims", "similarity", "scores"]
    for k in candidate_keys:
        if isinstance(out, dict) and (k in out):
            scores = out[k]
            break
    else:
        scores = out if torch.is_tensor(out) else None
    if scores is None:
        raise RuntimeError("VisionTextSigLIP forward 결과에서 점수 행렬 키를 찾지 못했다.")
    return scores
def _topk_unique_by_label(
    scores_1d,                  # shape [M], torch.Tensor or np.ndarray
    labels_of_texts: List[int], # 길이 M
    k: int,
    min_score: Optional[float] = None,  # (옵션) 임계점수, 없으면 전체 후보 고려
):
    """
    점수 상위부터 훑으며 '라벨이 중복되지 않도록' 최대 k개 선택한다.
    반환: chosen_idx(list[int]), chosen_scores(list[float])
    """
    if torch.is_tensor(scores_1d):
        scores_np = scores_1d.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores_1d, dtype=np.float32)

    order = np.argsort(-scores_np)  # 내림차순
    chosen_idx, chosen_scores = [], []
    seen_labels = set()

    for j in order:
        s = float(scores_np[j])
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
def run_infer(
    ckpt_tag: str = "last",
    topk: int = 5,
    text_batch: int = 256,
    image_paths: Optional[List[str]] = None,
    max_demo_images: int = 8,
    vt_ckpt_path: Optional[str] = None,
    token_ckpt_path: Optional[str] = None,
    strict_load: bool = False,
    fused_outdir: Optional[str] = None,   # ⬅️ 전체 npy 저장 경로 지정 가능
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vt, token_model = build_model(device, vt_ckpt_path=vt_ckpt_path, token_ckpt_path=token_ckpt_path, strict_load=strict_load)
    if not vt_ckpt_path and not token_ckpt_path:
        load_train_state_or_pair(vt, token_model, tag=ckpt_tag)
    vt.eval()

    assert os.path.isfile(ITEMS_TXT_PATH), f"not found: {ITEMS_TXT_PATH}"
    items_txt = load_jsonl(ITEMS_TXT_PATH)
    texts = [it["text"] for it in items_txt]
    labels_of_texts = [int(it["label"]) for it in items_txt]
    print(f"[Infer] #texts (candidates): {len(texts)}")

    label2inds: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels_of_texts):
        label2inds[lab].append(idx)

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

    # 점수/임베딩 버퍼
    all_scores = torch.empty((images.size(0), len(texts)), dtype=torch.float32, device=device)
    proj_dim = vt.proj_out_dim if hasattr(vt, "proj_out_dim") else 1024
    text_embs_all = np.zeros((len(texts), proj_dim), dtype=np.float32)

    collate = TextCollatorSingle(vt.text_encoder.tokenizer, max_length=128)

    start = 0
    while start < len(texts):
        end = min(start + text_batch, len(texts))
        cur_items = [{"text": t, "label": 0} for t in texts[start:end]]
        batch = collate(cur_items)

        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)

        use_amp = (device == "cuda")
        amp_dtype = torch.bfloat16
        ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp)

        with torch.no_grad():
            with ctx:
                scores = _forward_scores(vt, images, input_ids, attention_mask)  # [B, cur]
                t_emb = vt.encode_texts(input_ids, attention_mask)               # [cur, D]
            text_embs_all[start:end, :] = t_emb.detach().cpu().to(torch.float32).numpy()

        all_scores[:, start:end] = scores
        start = end

    # ============================================== #
    #   ✅ 모든 이미지: '라벨 중복 없이' Top-K(양수 임계) 가중 평균
    # ============================================== #
    fused_vecs = []
    meta_all = {}

    for i, img_path in enumerate(tqdm(valid_paths, desc="Processing")):
        scores_i = all_scores[i]  # torch.Tensor [M]
        chosen_idx, weights = _topk_unique_by_label(
            scores_1d=scores_i,
            labels_of_texts=labels_of_texts,
            k=topk,
            min_score=0.0,   # 0보다 큰(=양수) 점수만 채택
        )

        if len(chosen_idx) == 0:
            print(f"[Fuse] skip (no positive scores): {img_path}")
            continue

        W = np.asarray(weights, dtype=np.float32)
        V = text_embs_all[chosen_idx, :]
        W = W / (W.sum() + 1e-12)
        fused_vec = (W[:, None] * V).sum(axis=0).astype(np.float32)
        fused_vecs.append(fused_vec)

        meta_all[os.path.basename(img_path)] = [
            {
                "rank": r + 1,
                "text_index": int(ti),
                "label": int(labels_of_texts[ti]),
                "score": float(weights[r]),
                "text": texts[ti],
            }
            for r, ti in enumerate(chosen_idx)
        ]

    fused_vecs = np.stack(fused_vecs, axis=0)  # [num_valid_images, D]

    # 저장 경로
    if fused_outdir is None or len(str(fused_outdir).strip()) == 0:
        fused_outdir = os.path.join(os.path.dirname(valid_paths[0]), "fused_vecs")
    os.makedirs(fused_outdir, exist_ok=True)

    npy_path = os.path.join(fused_outdir, "all_images_fused_topk_unique.npy")
    np.save(npy_path, fused_vecs)

    meta_path = os.path.join(fused_outdir, "all_images_fused_topk_unique.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    print(f"[Fuse] saved one npy for all images: {npy_path} (shape={fused_vecs.shape})")
    print(f"[Fuse] saved meta json: {meta_path}")


# ---------------------
# Infer: JSONL → 임베딩 → 검색/정렬 → log_print
# --------------------- 

def _extract_name_by_regex(p: str) -> str:
    m = re.search(r"/([^/]+)\.", p)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(p))[0]

def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = torch.nn.functional.normalize(A, dim=1)
    B = torch.nn.functional.normalize(B, dim=1)
    return A @ B.t()

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
    ap.add_argument("--text_batch", type=int, default=256, help="추론 시 텍스트 배치 크기")
    ap.add_argument("--infer_images", type=str, default="",
                    help="추론용 이미지 경로(쉼표/공백 구분) 또는 글롭 패턴. 미지정 시 items_img.jsonl 일부로 데모")
    ap.add_argument("--max_demo_images", type=int, default=500, help="infer_images 미지정 시 데모에 사용할 이미지 수")

    # JSONL 기반 매칭 입력
    ap.add_argument("--query_jsonl", type=str, default="",
                    help="(match) query 이미지 목록 JSONL 경로")
    ap.add_argument("--ref_jsonl",   type=str, default="",
                    help="(match) ref   이미지 목록 JSONL 경로")
    ap.add_argument("--desc_sim", type=int, default=1, help="1이면 유사도 내림차순 정렬")
    ap.add_argument("--log_topk", type=int, default=5, help="log_print에서 표시할 Top-K")

    # ▶ 통합(train_state_*) 체크포인트 재개 옵션(선택)
    ap.add_argument("--resume_path", type=str,
                    default="",
                    help="직접 지정한 train_state_*.pt 경로")
    ap.add_argument("--resume_tag", type=str, default="",
                    help="CKPT_DIR/train_state_{TAG}.pt 로드")
    ap.add_argument("--resume_all", type=int, default=0,
                    help="1이면 optimizer/scaler까지 함께 로드")

    # ▶ vt/token 가중치를 “각각” 로드하기 위한 경로(학습/추론 공통)
    ap.add_argument("--vt_ckpt_path", type=str, default="/root/project/llm_prompt/llm_machine/checkpoint_siglip/vt_epoch105_final_multimodal.pth", help="vt_*.pt 등 개별 가중치 파일 경로")
    ap.add_argument("--token_ckpt_path", type=str, default="/root/project/llm_prompt/llm_machine/checkpoint_siglip/token_epoch105_final_multimodal.pth", help="token_*.pt 등 개별 가중치 파일 경로")
    ap.add_argument("--strict_load", type=int, default=0, help="1이면 state_dict strict 로드")

    # ▶ 저장 정책: 통합 train_state 저장 여부
    ap.add_argument("--save_full_state", type=int, default=1,
                    help="1이면 train_state_*.pt도 함께 저장, 0이면 vt/token만 저장")
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
        # 기존 데모 추론 경로 (ITEMS_IMG_PATH 기반)
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
        )
