# infer.py
import os, json, argparse, glob
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from main import (
    build_model, load_jsonl, build_eval_transform, load_images_as_tensor,
    forward_scores, topk_unique_by_label, DEFAULT_IMAGE_SIZE, DEFAULT_NORM_MEAN, DEFAULT_NORM_STD,
)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def gather_image_paths(image_glob: str) -> List[str]:
    raw: List[str] = []
    for token in image_glob.replace(",", " ").split():
        if any(ch in token for ch in "*?[]"):      # 와일드카드
            raw.extend(glob.glob(token))
        elif os.path.isdir(token):                 # 폴더
            for ext in IMG_EXTS:
                raw.extend(glob.glob(os.path.join(token, f"*{ext}")))
        else:                                      # 파일
            raw.append(token)
    # 파일 + 확장자 필터
    return [p for p in raw if os.path.isfile(p) and p.lower().endswith(IMG_EXTS)]

def chunk_indices(n: int, bs: int) -> List[Tuple[int,int]]:
    return [(i, min(i+bs, n)) for i in range(0, n, bs)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items_txt", type=str, required=True, help="텍스트 후보 JSONL")
    ap.add_argument("--image_glob", type=str, required=True, help="추론 이미지 글롭/경로(쉼표/공백 가능)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--text_batch", type=int, default=32)
    ap.add_argument("--image_batch", type=int, default=16)   # ✅ 추가: 이미지 배치 크기
    ap.add_argument("--fused_outdir", type=str, default="")

    ap.add_argument("--vt_ckpt_path", type=str, default="")
    ap.add_argument("--token_ckpt_path", type=str, default="")
    ap.add_argument("--strict_load", type=int, default=0)

    # (옵션) pair 로딩용 태그/디렉토리 지원
    ap.add_argument("--ckpt_dir", type=str, default="")
    ap.add_argument("--ckpt_tag", type=str, default="latest")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vt, token_model = build_model(
        device,
        vt_ckpt_path=(args.vt_ckpt_path or None),
        token_ckpt_path=(args.token_ckpt_path or None),
        strict_load=bool(args.strict_load),
    )

    # 가중치 로드: 개별 경로 없으면 train_state/pair에서 불러오기
    if (not args.vt_ckpt_path) and (not args.token_ckpt_path) and args.ckpt_dir:
        state_path = os.path.join(args.ckpt_dir, f"train_state_{args.ckpt_tag}.pt")
        if os.path.isfile(state_path):
            raw = torch.load(state_path, map_location="cpu")
            if "vt" in raw: vt.load_state_dict(raw["vt"], strict=False)
            if "token" in raw:
                (token_model.module if hasattr(token_model, "module") else token_model).load_state_dict(raw["token"], strict=False)
            print(f"[Load] from train_state: {state_path}")
        else:
            vt_path = os.path.join(args.ckpt_dir, f"vt_{args.ckpt_tag}.pt")
            token_path = os.path.join(args.ckpt_dir, f"token_{args.ckpt_tag}.pt")
            if os.path.isfile(vt_path):
                vt.load_state_dict(torch.load(vt_path, map_location="cpu"), strict=False)
            if os.path.isfile(token_path):
                (token_model.module if hasattr(token_model, "module") else token_model).load_state_dict(
                    torch.load(token_path, map_location="cpu"), strict=False
                )
            print(f"[Load] from pair: {vt_path} , {token_path}")

    vt.eval()

    # 텍스트 후보 읽기
    items_txt = load_jsonl(args.items_txt)
    texts = [it["text"] for it in items_txt]
    labels_of_texts = [int(it["label"]) for it in items_txt]
    M = len(texts)
    print(f"[Infer] #texts (candidates): {M}")

    # 이미지 경로 수집
    image_paths = gather_image_paths(args.image_glob)
    if not image_paths:
        raise SystemExit("image_glob에 해당하는 이미지가 없다.")
    N = len(image_paths)
    print(f"[Infer] #images: {N}")

    # 변환
    tfm = build_eval_transform(
        image_size=DEFAULT_IMAGE_SIZE,
        norm_mean=tuple(DEFAULT_NORM_MEAN),
        norm_std=tuple(DEFAULT_NORM_STD)
    )

    # 1) 텍스트 임베딩만 선계산(메모리에 저장) → [M, D]
    proj_dim = getattr(vt, 'proj_out_dim', 1024)
    text_embs_all = np.zeros((M, proj_dim), dtype=np.float32)

    from llm_machine.data_linked import TextCollatorSingle
    collate = TextCollatorSingle(vt.text_encoder.tokenizer, max_length=128)

    use_amp = (device == "cuda")
    amp_dtype = torch.bfloat16
    autocast_ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp)

    start = 0
    while start < M:
        end = min(start + args.text_batch, M)
        cur_items = [{"text": t, "label": 0} for t in texts[start:end]]
        batch = collate(cur_items)

        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        with torch.no_grad():
            with autocast_ctx:
                t_emb = vt.encode_texts(input_ids, attention_mask)  # [cur, D]
        text_embs_all[start:end] = t_emb.detach().cpu().to(torch.float32).numpy()
        start = end
    # 텍스트 쪽은 끝.

    # 2) 이미지 배치 × 텍스트 배치로 점수 계산 후 즉시 Top-K 융합
    fused_vecs: List[np.ndarray] = []
    meta_all: Dict[str, Any] = {}

    for is_, ie in chunk_indices(N, args.image_batch):
        # 이미지 배치 적재
        imgs_tensor, valid_paths = load_images_as_tensor(image_paths[is_:ie], tfm)  # [B, C, H, W]
        if imgs_tensor is None or len(valid_paths) == 0:
            continue
        imgs_tensor = imgs_tensor.to(device, non_blocking=True)
        B = imgs_tensor.size(0)

        # 현재 이미지 배치에 대한 점수 버퍼 [B, M] (GPU 메모리 절약을 위해 float16/32 선택)
        scores_bm = torch.empty((B, M), dtype=torch.float32, device=device)

        # 텍스트 배치로 타일링
        t0 = 0
        while t0 < M:
            t1 = min(t0 + args.text_batch, M)
            cur_items = [{"text": texts[i], "label": 0} for i in range(t0, t1)]
            batch = collate(cur_items)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)

            with torch.no_grad():
                with autocast_ctx:
                    scores = forward_scores(vt, imgs_tensor, input_ids, attention_mask)  # [B, cur]
            scores_bm[:, t0:t1] = scores
            t0 = t1

        # 이미지 배치의 각 이미지에 대해 즉시 Top-K 융합 및 메타 생성
        for bi, img_path in enumerate(valid_paths):
            scores_i = scores_bm[bi]  # [M] on device
            # CPU로 이동하여 numpy로 가볍게 처리
            scores_i_np = scores_i.detach().cpu().to(torch.float32).numpy()

            chosen_idx, weights = topk_unique_by_label(
                scores_1d=scores_i_np,
                labels_of_texts=labels_of_texts,
                k=args.topk,
                min_score=0.0,
            )
            if len(chosen_idx) == 0:
                print(f"[Fuse] skip (no positive scores): {img_path}")
                continue

            W = np.asarray(weights, dtype=np.float32)
            V = text_embs_all[chosen_idx, :]                     # [K, D]
            W = W / (W.sum() + 1e-12)
            fused_vec = (W[:, None] * V).sum(axis=0).astype(np.float32)  # [D]
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

        # 배치 끝날 때 메모리 정리
        del imgs_tensor, scores_bm
        torch.cuda.empty_cache() if device == "cuda" else None

    if not fused_vecs:
        raise RuntimeError("fused_vecs가 비어 있음")

    fused_vecs = np.stack(fused_vecs, axis=0)

    # 저장
    # outdir 기본값: 첫 유효 이미지와 같은 폴더 내 fused_vecs/
    first_valid = next((p for p in image_paths if os.path.isfile(p)), None)
    base_dir = os.path.dirname(first_valid) if first_valid else os.getcwd()
    outdir = args.fused_outdir or os.path.join(base_dir, "fused_vecs")
    os.makedirs(outdir, exist_ok=True)

    np.save(os.path.join(outdir, "all_images_fused_topk_unique.npy"), fused_vecs)
    with open(os.path.join(outdir, "all_images_fused_topk_unique.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    print(f"[Fuse] saved npy/meta to: {outdir}")

if __name__ == "__main__":
    main()
