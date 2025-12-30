#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble_match_token_text.py
- Token 모델로 이미지 임베딩(.npy) 생성/저장
- VisionTextSigLIP(+LLMTextEncoder)로 문양 Top-K(sum=10) 임베딩(.npy) 생성/저장
- 두 임베딩을 0.9(img) : 0.1(text) 비율로 가중 합 → 정규화 → 코사인 "거리"(1 - cos) 행렬 계산
- 정렬 후 log_print 호출까지 수행

필수 모듈(사용자 환경 가정):
- networks.Token
- llm_machine (LLMTextEncoder, VisionTextSigLIP, data_linked.TextCollatorSingle)
- logprint.log_print
"""

import os, re, csv, json, time, argparse
import os.path as osp
from glob import glob
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# ==== 사용자 환경 모듈 ====
from networks import Token
from llm_machine.logprint import log_print
from llm_machine import LLMTextEncoder, VisionTextSigLIP
from llm_machine.data_linked import TextCollatorSingle

# ---------------------
# 공통 설정
# ---------------------
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD  = [0.229, 0.224, 0.225]
TOKEN_IMG_SIZE = (1024,1024)    # Token 전처리
VT_IMG_SIZE    = (1024, 512)    # VisionText 전처리
PROJ_DIM = 1024
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# ---------------------
# 유틸
# ---------------------
def set_seed(seed: int = 11):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _debug_cuda(prefix=""):
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        alloc = torch.cuda.memory_allocated(0) / 1024**2
        rsv   = torch.cuda.memory_reserved(0) / 1024**2
        print(f"[CUDA] {prefix} | {device_name} | alloc={alloc:.1f}MB reserved={rsv:.1f}MB")
    else:
        print(f"[CUDA] {prefix} | CPU mode")

def _extract_name_by_regex(p: str) -> str:
    m = re.search(r"/([^/]+)\.", p)
    return m.group(1) if m else osp.splitext(osp.basename(p))[0]

def cosine_distance_matrix(a: torch.Tensor, b: torch.Tensor, batch_q: int = 256, batch_r: int = 4096) -> torch.Tensor:
    """
    return: 1 - cosine_similarity in [Nq, Nr]
    """
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    out = torch.empty((a.size(0), b.size(0)), device=a.device, dtype=a.dtype)
    for i in range(0, a.size(0), batch_q):
        ai = a[i:i+batch_q]
        for j in range(0, b.size(0), batch_r):
            bj = b[j:j+batch_r]
            sims = ai @ bj.t()
            out[i:i+batch_q, j:j+batch_r] = 1.0 - sims
    return out

# ---------------------
# 이미지 로더 (공통)
# ---------------------
def pil_loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")

class ImageFromList(data.Dataset):
    def __init__(self, Image_paths: List[str], transforms: T.Compose, imsize: Tuple[int, int]):
        super().__init__()
        self.Image_paths = Image_paths
        self.transforms = transforms
        self.imsize = imsize

    def __getitem__(self, index):
        path = self.Image_paths[index]
        img  = pil_loader(path)
        img  = T.Resize(self.imsize, interpolation=InterpolationMode.BICUBIC, antialias=True)(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.Image_paths)

# ---------------------
# 1) Token 임베딩 추출/저장
# ---------------------
@torch.no_grad()
def extract_token_vectors(
    image_paths: List[str],
    token_ckpt: str,
    device: str,
    batch_size: int,
    num_workers: int,
    save_path: str,
):
    if osp.exists(save_path):
        try:
            np.load(save_path, mmap_mode="r")
            print(f"[Token] skip exists: {save_path}")
            return save_path
        except Exception:
            print("[Token] existing npy unreadable → re-create")

    model = Token(mode='infer').to(device)
    print(f"[Token] load: {token_ckpt}")
    state = torch.load(token_ckpt, map_location='cpu', weights_only=False)
    # state_dict일 수도 있고 wrapper일 수도 있어 관용 처리
    sd = state.get('state_dict', state)
    msg = model.load_state_dict(sd, strict=False)
    print(f"[Token] load_state_dict: {msg}")

    model.eval()
    tfm = T.Compose([T.ToTensor(), T.Normalize(IMG_NORM_MEAN, IMG_NORM_STD)])
    loader = DataLoader(
        ImageFromList(image_paths, transforms=tfm, imsize=TOKEN_IMG_SIZE),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    vecs = np.zeros((len(image_paths), model.outputdim), dtype=np.float32)
    p = 0
    for imgs in tqdm(loader, desc="[Token] encode", ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        out  = model.forward_test(imgs)           # [B, D]
        v    = out.detach().float().cpu().numpy()
        b    = v.shape[0]
        vecs[p:p+b, :] = v
        p += b
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    np.save(save_path, vecs)
    print(f"[Token] saved -> {save_path} shape={vecs.shape}")
    return save_path

# ---------------------
# 2) 문양 Top-K(sum=10) 임베딩 추출/저장 (VisionTextSigLIP)
# ---------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

@torch.no_grad()
def precompute_text_embeddings(vt, texts: List[str], device: str, text_batch: int, debug: bool=False) -> np.ndarray:
    collate = TextCollatorSingle(vt.text_encoder.tokenizer, max_length=128)
    use_amp = (device.startswith("cuda") and torch.cuda.is_available())
    try:
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    except Exception:
        amp_dtype = torch.float16

    embs = np.zeros((len(texts), PROJ_DIM), dtype=np.float32)
    s = 0; t0 = time.perf_counter()
    while s < len(texts):
        e = min(s + text_batch, len(texts))
        cur = [{"text": t, "label": 0} for t in texts[s:e]]
        batch = collate(cur)
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = vt.encode_texts(input_ids, attention_mask)  # [cur, D]
        embs[s:e, :] = out.detach().float().cpu().numpy()
        s = e
        if debug: _debug_cuda(f"text {e}/{len(texts)}")
    return embs

def topk_unique_pos_by_label(
    scores: np.ndarray,           # [Nt]
    labels_of_texts: List[int],   # len=Nt
    k: int
) -> Tuple[List[int], List[float]]:
    idx_sorted = np.argsort(-scores)  # 점수 내림차순
    seen = set()
    out_idx, out_scores = [], []

    Nt = len(scores)
    for i in idx_sorted:
        s = float(scores[i])
        if not np.isfinite(s) or s <= 0.0:
            continue
        lab = labels_of_texts[i] if i < len(labels_of_texts) else -1
        # 라벨이 없거나 음수면 해당 텍스트 자체를 유니크 키로 취급
        key = lab if (isinstance(lab, int) and lab >= 0) else f"__nolabel_{i}"
        if key in seen:
            continue
        seen.add(key)
        out_idx.append(int(i))
        out_scores.append(s)
        if len(out_idx) >= k:
            break
    return out_idx, out_scores


def build_vt_model(device: str, vt_ckpt: Optional[str], token_ckpt_for_vt: Optional[str]):
    # 토큰 백본 freeze (학습 아님, 추론)
    token_model = Token(outputdim=PROJ_DIM, classifier_num=81313, mode='train').to(device)
    for name, p in token_model.named_parameters():
        p.requires_grad = False if name.startswith("backbone") else True
    token_model.train()  # 내부에서 eval 모드로 쓰기 때문에 상관 없음

    text_encoder = LLMTextEncoder(
        model_name=MODEL_NAME, device=device, dtype=torch.bfloat16,
        train_llm=True, use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.1, pooling="mean",
    )
    vt = VisionTextSigLIP(
        token_model=token_model, text_encoder=text_encoder,
        vision_dim=PROJ_DIM, proj_out_dim=PROJ_DIM, temperature_init=0.06,
    ).to(device).eval()

    if vt_ckpt and osp.isfile(vt_ckpt):
        sd = torch.load(vt_ckpt, map_location="cpu")
        vt.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict(), strict=False)
        print(f"[VT] loaded vt: {vt_ckpt}")
    if token_ckpt_for_vt and osp.isfile(token_ckpt_for_vt):
        mdl = token_model.module if hasattr(token_model, "module") else token_model
        raw = torch.load(token_ckpt_for_vt, map_location="cpu")
        sd  = raw if isinstance(raw, dict) else raw.state_dict()
        filt = {k:v for k,v in sd.items() if k in mdl.state_dict() and mdl.state_dict()[k].shape == v.shape}
        miss = mdl.load_state_dict(filt, strict=False)
        print(f"[VT] partial load token({len(filt)} keys), miss={len(getattr(miss,'missing_keys',[]))}")
    return vt

@torch.no_grad()
def build_text_fused_embeddings_pos_top5_unique(
    image_paths: List[str],
    items_txt_path: str,
    vt_ckpt: Optional[str],
    token_ckpt_for_vt: Optional[str],
    device: str,
    image_batch: int,
    text_batch: int,
    save_path: str,
    save_meta_path: Optional[str] = None,
    debug: bool = False,
):
    """
    변경 사항:
    - 점수 > 0인 텍스트만 대상으로 함.
    - 라벨 중복 제거(같은 라벨은 1개만 선택).
    - 그 중 상위 5개 선택.
    - 선택된 텍스트 임베딩의 '단순 평균'(비가중)으로 fused 벡터 생성.
    - 메타에는 선택된 항목들만 기록.
    """
    if osp.exists(save_path):
        try:
            np.load(save_path, mmap_mode="r")
            print(f"[Text@pos_top5_unique] skip exists: {save_path}")
            return save_path
        except Exception:
            print("[Text@pos_top5_unique] existing npy unreadable → re-create")

    vt = build_vt_model(device, vt_ckpt, token_ckpt_for_vt)
    vt.eval()

    items_txt = load_jsonl(items_txt_path)
    texts = [it["text"] for it in items_txt]
    labels_of_texts = [int(it.get("label", -1)) for it in items_txt]
    text_embs = precompute_text_embeddings(vt, texts, device, text_batch, debug=debug)  # [Nt, D], 정규화 가정

    tfm = T.Compose([
        T.Resize(VT_IMG_SIZE, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(IMG_NORM_MEAN, IMG_NORM_STD),
    ])
    loader = DataLoader(
        ImageFromList(image_paths, transforms=tfm, imsize=VT_IMG_SIZE),
        batch_size=image_batch, shuffle=False, num_workers=4, pin_memory=True
    )

    collate = TextCollatorSingle(vt.text_encoder.tokenizer, max_length=128)
    use_amp = (device.startswith("cuda") and torch.cuda.is_available())
    try:
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    except Exception:
        amp_dtype = torch.float16

    fused = np.zeros((len(image_paths), PROJ_DIM), dtype=np.float32)
    meta_all: Dict[str, Any] = {}
    p = 0

    for imgs in tqdm(loader, desc="[Text@pos_top5_unique] score+select", ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        B = imgs.size(0)
        scores_full = np.empty((B, len(texts)), dtype=np.float32)

        # 텍스트 배치로 점수 계산
        s = 0
        while s < len(texts):
            e = min(s + text_batch, len(texts))
            cur = [{"text": t, "label": 0} for t in texts[s:e]]
            batch = collate(cur)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = vt(imgs, input_ids, attention_mask, targets=None)

            # 점수 키 탐색
            for k in ["logits", "logits_per_image", "sims", "similarity", "scores"]:
                if isinstance(out, dict) and (k in out):
                    scores = out[k]; break
            else:
                scores = out if torch.is_tensor(out) else None
            if scores is None:
                raise RuntimeError("VT forward 결과에서 점수 행렬을 찾을 수 없음.")

            scores_full[:, s:e] = scores.detach().float().cpu().numpy()
            s = e

        # 라벨 중복 제거 + 양수 상위 5 → 평균
        for bi in range(B):
            sc = scores_full[bi]
            sc = np.where(np.isfinite(sc), sc, -np.inf)

            idxs, raw_scores = topk_unique_pos_by_label(sc, labels_of_texts, k=5)

            if len(idxs) == 0:
                fused[p + bi, :] = 0.0
                if save_meta_path:
                    meta_all[str(p + bi)] = []
                continue

            V = text_embs[np.asarray(idxs), :]                      # [m, D]
            fused[p + bi, :] = V.mean(axis=0).astype(np.float32)    # 비가중 평균

            if save_meta_path:
                meta_all[str(p + bi)] = [
                    {
                        "rank": r + 1,
                        "text_index": int(ti),
                        "label": int(labels_of_texts[ti]),
                        "score_raw": float(raw_scores[r]),
                        "text": str(texts[ti]),
                    }
                    for r, ti in enumerate(idxs)
                ]

        p += B

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    np.save(save_path, fused)
    print(f"[Text@pos_top5_unique] saved -> {save_path} shape={fused.shape}")

    if save_meta_path:
        with open(save_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_all, f, ensure_ascii=False, indent=2)
        print(f"[Text@pos_top5_unique] saved meta -> {save_meta_path}")

    return save_path

# ---------------------
# 3) 앙상블 + 매칭 + log_print
# ---------------------
def run(
    query_dir: str,
    ref_dir: str,
    items_txt: str,
    token_ckpt_for_token: str,
    vt_ckpt: Optional[str],
    token_ckpt_for_vt: Optional[str],
    out_dir: str,
    weight_img: float = 0.9,
    weight_txt: float = 0.1,
    batch_token: int = 5,
    workers_token: int = 16,
    batch_text_img: int = 8,
    batch_text_txt: int = 512,
    topk_text: int = 3,
    require_positive: bool = True,
    save_meta: bool = False,
    testcsv: Optional[str] = None,
    desc_sim: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(11)
    os.makedirs(out_dir, exist_ok=True)

    # 이미지 경로
    ref_images   = sorted(glob(osp.join(ref_dir, "*")))
    query_images = sorted(glob(osp.join(query_dir, "*")))
    if not ref_images or not query_images:
        raise RuntimeError("query/ref 폴더에 이미지가 없습니다.")
    print(f"[Run] #query={len(query_images)}  #ref={len(ref_images)}  device={device}")

    # 이름 테이블
    ref_names   = [os.path.splitext(os.path.basename(x.strip()))[0] for x in ref_images]
    ref_indices = {name: i for i, name in enumerate(ref_names)}
    query_order = [os.path.splitext(os.path.basename(x.strip()))[0] for x in query_images]

    # 1) Token 임베딩
    token_query_npy = osp.join(out_dir, "query_token.npy")
    token_ref_npy   = osp.join(out_dir, "ref_token.npy")
    extract_token_vectors(query_images, token_ckpt_for_token, device, batch_token, workers_token, token_query_npy)
    extract_token_vectors(ref_images,   token_ckpt_for_token, device, batch_token, workers_token, token_ref_npy)

    # 2) Text Top-K(sum=10) 임베딩
    text_query_npy = osp.join(out_dir, f"query_text_top{topk_text}_sum10.npy")
    text_ref_npy   = osp.join(out_dir, f"ref_text_top{topk_text}_sum10.npy")
    meta_query = osp.join(out_dir, "query_text_topk.meta.json") if save_meta else None
    meta_ref   = osp.join(out_dir, "ref_text_topk.meta.json")   if save_meta else None

    build_text_fused_embeddings_pos_top5_unique(
        query_images, items_txt, vt_ckpt, token_ckpt_for_vt, device,
        image_batch=batch_text_img, text_batch=batch_text_txt, 
         save_path=text_query_npy, save_meta_path=meta_query, debug=False
    )
    build_text_fused_embeddings_pos_top5_unique(
        ref_images, items_txt, vt_ckpt, token_ckpt_for_vt, device,
        image_batch=batch_text_img, text_batch=batch_text_txt, 
         save_path=text_ref_npy, save_meta_path=meta_ref, debug=False
    )

    # 3) 앙상블 (가중 합 → 정규화)
    QT = torch.from_numpy(np.load(token_query_npy)).float().to(device)
    RT = torch.from_numpy(np.load(token_ref_npy)).float().to(device)
    QX = torch.from_numpy(np.load(text_query_npy)).float().to(device)
    RX = torch.from_numpy(np.load(text_ref_npy)).float().to(device)

    # 각 벡터 정규화 후 가중 합
    def _norm(x): return F.normalize(x, p=2, dim=1)
    Q_ens = _norm(weight_img * _norm(QT) + weight_txt * _norm(QX))
    R_ens = _norm(weight_img * _norm(RT) + weight_txt * _norm(RX))
    # 4) 코사인 "거리"(1 - cos) 행렬 → 정렬
    D = cosine_distance_matrix(Q_ens, R_ens)   # 낮을수록 유사
    sorted_distances, sorted_indices = torch.sort(D, dim=1, descending=False)

    # 5) log_print 호출 (사용자 기존 함수 시그니처 유지)
    #    log_print(sorted_indices, sorted_distances, query_order, ref_indices, ref_names, args)
    class _Args:
        # 필요 시 확장
        def __init__(self, testcsv=None, desc_sim=True):
            self.testcsv = testcsv
            self.desc_sim = desc_sim
    args_obj = _Args(testcsv=testcsv, desc_sim=desc_sim)

    print(f"[Out] call log_print ...")
    log_print(sorted_indices, sorted_distances, query_order, ref_indices, ref_names, args_obj)

    # (선택) 상위 매칭 CSV 저장
    out_csv = osp.join(out_dir, "ensemble_match_top5.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_name", "ref_name", "distance(1-cos)"])
        topN = 5
        for qi, qname in enumerate(query_order):
            inds = sorted_indices[qi, :topN].tolist()
            dists = sorted_distances[qi, :topN].tolist()
            for ri, dist in zip(inds, dists):
                w.writerow([qname, ref_names[ri], float(dist)])
    print(f"[Out] saved -> {out_csv}")

# ---------------------
# CLI
# ---------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', type=str, required=True, help='query 이미지 폴더')
    ap.add_argument('--ref',   type=str, required=True, help='ref   이미지 폴더')
    # Token
    ap.add_argument('--token_ckpt_for_token', type=str, required=True, help='Token 모델 가중치(pth, state_dict 포함)')
    ap.add_argument('--batch_token', type=int, default=64)
    ap.add_argument('--workers_token', type=int, default=16)

    # Text Top-K(sum10)
    ap.add_argument('--items_txt', type=str, required=True, help='텍스트 후보 JSONL (fields: text, label)')
    ap.add_argument('--vt_ckpt',   type=str, default='/root/project/llm_prompt/llm_machine/checkpoint_siglip/vt_epoch056.pt', help='VisionTextSigLIP 가중치(선택)')
    ap.add_argument('--token_ckpt_for_vt', type=str, default='/root/project/llm_prompt/llm_machine/checkpoint_siglip/token_epoch056.pt', help='VT 내부 Token 부분에 부분 로드할 가중치(선택)')
    ap.add_argument('--batch_text_img', type=int, default=8, help='문양 임베딩용 이미지 배치')
    ap.add_argument('--batch_text_txt', type=int, default=512, help='문양 임베딩용 텍스트 배치')
    ap.add_argument('--topk_text', type=int, default=5)
    ap.add_argument('--require_positive', type=int, default=1)
    # Ensemble & Out
    ap.add_argument('--w_img', type=float, default=0.7)
    ap.add_argument('--w_txt', type=float, default=0.3)
    ap.add_argument('--out_dir', type=str, default='/root/project/embeddings')
    ap.add_argument('--save_meta', type=int, default=1)
    ap.add_argument('--testcsv', type=str, default='')
    ap.add_argument('--desc_sim', type=int, default=1)
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(
        query_dir=args.query,
        ref_dir=args.ref,
        items_txt=args.items_txt,
        token_ckpt_for_token=args.token_ckpt_for_token,
        vt_ckpt=(args.vt_ckpt if args.vt_ckpt else None),
        token_ckpt_for_vt=(args.token_ckpt_for_vt if args.token_ckpt_for_vt else None),
        out_dir=args.out_dir,
        weight_img=args.w_img,
        weight_txt=args.w_txt,
        batch_token=args.batch_token,
        workers_token=args.workers_token,
        batch_text_img=args.batch_text_img,
        batch_text_txt=args.batch_text_txt,
        topk_text=args.topk_text,
        require_positive=bool(args.require_positive),
        save_meta=bool(args.save_meta),
        testcsv=(args.testcsv if args.testcsv else None),
        desc_sim=bool(args.desc_sim),
    )
