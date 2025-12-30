import os, re, glob, json, time, random
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode

# ---- 기본 상수 ----
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_IMAGE_SIZE = (1024, 512)
DEFAULT_NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
DEFAULT_NORM_STD  = [0.26862954, 0.26130258, 0.27577711]

# =============================
# 파일/경로/로더
# =============================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def gather_paths(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    parts: List[str] = []
    for token in arg.replace(",", " ").split():
        if any(ch in token for ch in "*?[]"):
            parts.extend(glob.glob(token))
        else:
            parts.append(token)
    out, seen = [], set()
    for p in parts:
        p = os.path.abspath(p)
        if (p not in seen) and os.path.isfile(p):
            out.append(p); seen.add(p)
    return out

def extract_name_by_regex(p: str) -> str:
    m = re.search(r"/([^/]+)\.", p)
    return m.group(1) if m else os.path.splitext(os.path.basename(p))[0]

def load_image_paths_from_jsonl(path: str) -> List[str]:
    assert os.path.isfile(path), f"jsonl not found: {path}"
    rows = load_jsonl(path)
    out = [r["image_path"] for r in rows if isinstance(r.get("image_path"), str) and r["image_path"]]
    if not out:
        raise RuntimeError(f"no image_path in {path}")
    return out

def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

# =============================
# 전처리/변환
# =============================
def build_train_transform(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    norm_mean: Tuple[float, float, float] = tuple(DEFAULT_NORM_MEAN),
    norm_std: Tuple[float, float, float] = tuple(DEFAULT_NORM_STD),
    white_bg_fill: bool = True,
    allow_flip: bool = False,
) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.RandomApply([
            T.RandomAffine(
                degrees=2, translate=(0.01, 0.01), scale=(0.98, 1.02), shear=1,
                interpolation=InterpolationMode.BICUBIC, fill=255 if white_bg_fill else 0
            )
        ], p=0.30),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.02, p=1.0)], p=0.10),
        T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05)], p=0.30),
        T.RandomHorizontalFlip(p=0.15 if allow_flip else 0.0),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.15),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

def build_eval_transform(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    norm_mean: Tuple[float, float, float] = tuple(DEFAULT_NORM_MEAN),
    norm_std: Tuple[float, float, float] = tuple(DEFAULT_NORM_STD),
) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

def load_images_as_tensor(paths: List[str], tfm: T.Compose) -> Tuple[torch.Tensor, List[str]]:
    image_tensors, valid_paths = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                image_tensors.append(tfm(im))
                valid_paths.append(p)
        except Exception as e:
            print(f"[Warn] 이미지 열기 실패: {p} ({e})")
    if not image_tensors:
        raise RuntimeError("유효한 이미지가 없습니다.")
    return torch.stack(image_tensors, dim=0), valid_paths

# =============================
# 텍스트/라벨
# =============================
from llm_machine.data_linked import TextCollatorSingle

def build_label2texts(items_txt: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    table: Dict[int, List[str]] = defaultdict(list)
    for it in items_txt:
        table[int(it["label"])].append(it["text"])
    return table

def unique_labels_in_batch(batch_img) -> List[int]:
    if hasattr(batch_img, "label_sets"):
        uniq = set()
        for labs in batch_img.label_sets:
            uniq.update(map(int, labs))
        return sorted(uniq)
    if isinstance(batch_img, dict) and "labels" in batch_img:
        labels = batch_img["labels"]
        if torch.is_tensor(labels) and labels.dim() == 2:
            uniq = set()
            for i in range(labels.size(0)):
                idxs = labels[i].nonzero(as_tuple=False).squeeze(1).tolist()
                uniq.update(map(int, idxs))
            return sorted(uniq)
        uniq = set()
        for labs in labels:
            uniq.update(map(int, labs))
        return sorted(uniq)
    raise TypeError("batch_img에서 라벨 정보를 찾을 수 없다. (label_sets 또는 labels 필요)")

def build_text_batch_one_per_label(batch_img, label2texts: Dict[int, List[str]], tokenizer, max_length: int = 128):
    uniq_labels = unique_labels_in_batch(batch_img)
    items, label_ids = [], []
    for lab in uniq_labels:
        cand = label2texts.get(lab, [])
        if not cand:
            continue
        txt = random.choice(cand)
        items.append({"text": txt, "label": int(lab)})
        label_ids.append(int(lab))
    if not items:
        return None
    collate = TextCollatorSingle(tokenizer, max_length=max_length)
    batch_txt = collate(items)
    setattr(batch_txt, "label_ids", label_ids)
    return batch_txt

# =============================
# 디버깅(옵션)
# =============================
class GradProbe:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.reset()
    def attach(self):
        self.detach()
        for _, p in self.model.named_parameters():
            if p.requires_grad:
                self._handles.append(p.register_hook(self._hook))
    def detach(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles.clear()
    def reset(self):
        self.had_any_grad = False
        self.grad_param_count = 0
        self.total_grad_norm_sq = 0.0
    def _hook(self, grad: torch.Tensor):
        if grad is None: return
        self.had_any_grad = True
        self.grad_param_count += 1
        with torch.no_grad():
            self.total_grad_norm_sq += float(grad.detach().pow(2).sum().cpu().item())
    def summary(self) -> Dict[str, Any]:
        return {
            "had_any_grad": bool(self.had_any_grad),
            "grad_param_count": int(self.grad_param_count),
            "grad_total_norm": float(self.total_grad_norm_sq ** 0.5),
        }

# =============================
# 스코어/메트릭 (공용 유틸)
# =============================
def topk_unique_by_label(scores_1d, labels_of_texts: List[int], k: int, min_score: Optional[float] = None):
    scores_np = scores_1d.detach().cpu().numpy() if torch.is_tensor(scores_1d) else np.asarray(scores_1d, dtype=np.float32)
    order = np.argsort(-scores_np)
    chosen_idx, chosen_scores, seen = [], [], set()
    for j in order:
        s = float(scores_np[j])
        if (min_score is not None) and (s < min_score): break
        lab = int(labels_of_texts[j])
        if lab in seen: continue
        seen.add(lab)
        chosen_idx.append(int(j))
        chosen_scores.append(s)
        if len(chosen_idx) >= k: break
    return chosen_idx, chosen_scores

def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = torch.nn.functional.normalize(A, dim=1)
    B = torch.nn.functional.normalize(B, dim=1)
    return A @ B.t()

# =============================
# 모델/체크포인트
# =============================
from llm_machine import LLMTextEncoder, VisionTextSigLIP
from networks import Token

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

def _unwrap_state_dict(maybe_wrapped: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for key in ("state_dict", "model", "net", "module"):
        if key in maybe_wrapped and isinstance(maybe_wrapped[key], dict):
            return maybe_wrapped[key]
    return maybe_wrapped

def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def _select_by_prefix_and_shape(checkpoint: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor], allowed_prefixes=("backbone", "tr")):
    filtered = {}
    for k, v in checkpoint.items():
        if not any(k.startswith(pfx) for pfx in allowed_prefixes):
            continue
        if (k in model_state) and (model_state[k].shape == v.shape):
            filtered[k] = v
    return filtered

def load_vt_from_path(vt: torch.nn.Module, path: str, map_location="cpu", strict: bool = False):
    assert os.path.isfile(path), f"vt ckpt not found: {path}"
    raw = _safe_torch_load(path, map_location=map_location)
    sd = raw if (isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values())) else _unwrap_state_dict(raw)
    sd = _strip_prefix_if_present(sd, "module.")
    vt.load_state_dict(sd, strict=strict)
    print(f"[LoadVT] loaded vt weights from {path}")

def load_token_from_path(token_model: torch.nn.Module, path: str, map_location="cpu", strict: bool = False):
    assert os.path.isfile(path), f"token ckpt not found: {path}"
    raw = _safe_torch_load(path, map_location=map_location)
    sd = raw if (isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values())) else _unwrap_state_dict(raw)
    sd = _strip_prefix_if_present(sd, "module.")
    model_wo_ddp = token_model.module if hasattr(token_model, "module") else token_model
    model_dict = model_wo_ddp.state_dict()
    filtered = _select_by_prefix_and_shape(sd, model_dict, allowed_prefixes=("backbone", "tr"))
    if len(filtered) == 0:
        filtered = {k: v for k, v in sd.items()
                    if (k in model_dict) and (model_dict[k].shape == v.shape)
                    and not any(x in k for x in ["classifier", "fc", "head", "heads", "arcface"])}
    incompatible = model_wo_ddp.load_state_dict(filtered, strict=False if not strict else True)
    print(f"[LoadToken] loaded {len(filtered)} keys from {path}")
    miss = list(getattr(incompatible, "missing_keys", []))
    unexp = list(getattr(incompatible, "unexpected_keys", []))
    if miss:  print(f"[LoadToken] Missing keys: {len(miss)} (first 10): {miss[:10]}")
    if unexp: print(f"[LoadToken] Unexpected keys: {len(unexp)} (first 10): {unexp[:10]}")

def build_model(device: str, vt_ckpt_path: Optional[str] = None, token_ckpt_path: Optional[str] = None, strict_load: bool = False):
    token_model = Token(outputdim=1024, classifier_num=81313, mode='train').to(device)
    for name, p in token_model.named_parameters():
        p.requires_grad = not (name.startswith("backbone") or name.startswith("tr"))
    token_model.train()

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
        token_model=token_model,
        text_encoder=text_encoder,
        vision_dim=1024,
        proj_out_dim=1024,
        temperature_init=0.06,
    ).to(device).train()

    if vt_ckpt_path:
        try:
            load_vt_from_path(vt, vt_ckpt_path, map_location="cpu", strict=strict_load)
        except Exception as e:
            print(f"[VT] Failed to load VT: {e}")
    if token_ckpt_path:
        try:
            load_token_from_path(token_model, token_ckpt_path, map_location="cpu", strict=strict_load)
        except Exception as e:
            print(f"[Token] Failed to load TOKEN: {e}")
    return vt, token_model

def save_weights(vt, token_model, optimizer, scaler, epoch, global_step, ckpt_dir: str, tag="last", save_full_state=True):
    os.makedirs(ckpt_dir, exist_ok=True)
    vt_sd = vt.state_dict()
    token_sd = (token_model.module if hasattr(token_model, "module") else token_model).state_dict()
    torch.save(vt_sd, os.path.join(ckpt_dir, f"vt_{tag}.pt"))
    torch.save(token_sd, os.path.join(ckpt_dir, f"token_{tag}.pt"))
    if save_full_state:
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "vt": vt_sd,
                "token": token_sd,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
            os.path.join(ckpt_dir, f"train_state_{tag}.pt"),
        )
