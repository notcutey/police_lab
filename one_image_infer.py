#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_one_demo.py
- 단일 이미지 1장에 대해 텍스트 후보 점수를 계산하고
  라벨 중복 없이 Top-K만 JSON으로 저장한다.
"""

import os, json, argparse, torch, numpy as np
from typing import Dict, Any, List

from main import (
    build_model, load_jsonl, build_eval_transform, load_images_as_tensor,
    forward_scores, topk_unique_by_label,
    DEFAULT_IMAGE_SIZE, DEFAULT_NORM_MEAN, DEFAULT_NORM_STD,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items_txt", type=str, required=True, help="텍스트 후보 JSONL 경로")
    ap.add_argument("--image_path", type=str, required=True, help="단일 이미지 파일 경로")
    ap.add_argument("--topk", type=int, default=10, help="라벨 중복 없이 상위 K")
    ap.add_argument("--text_batch", type=int, default=32, help="텍스트 배치 크기")
    ap.add_argument("--out_json", type=str, default="infer_one_topk.json", help="저장할 JSON 경로")

    ap.add_argument("--vt_ckpt_path", type=str, default="", help="VisionText 가중치")
    ap.add_argument("--token_ckpt_path", type=str, default="", help="Token 가중치(필요시)")
    ap.add_argument("--strict_load", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isfile(args.image_path):
        raise SystemExit(f"이미지 파일이 존재하지 않음: {args.image_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 로드
    vt, token_model = build_model(
        device,
        vt_ckpt_path=(args.vt_ckpt_path or None),
        token_ckpt_path=(args.token_ckpt_path or None),
        strict_load=bool(args.strict_load),
    )
    vt.eval()

    # 텍스트 후보 로드
    items_txt = load_jsonl(args.items_txt)
    texts: List[str] = [it["text"] for it in items_txt]
    labels_of_texts: List[int] = [int(it["label"]) for it in items_txt]
    M = len(texts)
    print(f"[Demo] #texts = {M}")

    # 이미지 로드(1장)
    tfm = build_eval_transform(
        image_size=DEFAULT_IMAGE_SIZE,
        norm_mean=tuple(DEFAULT_NORM_MEAN),
        norm_std=tuple(DEFAULT_NORM_STD),
    )
    imgs_tensor, valid_paths = load_images_as_tensor([args.image_path], tfm)
    if imgs_tensor is None or len(valid_paths) == 0:
        raise SystemExit("이미지 로드 실패")
    imgs_tensor = imgs_tensor.to(device, non_blocking=True)  # [1, C, H, W]

    # 텍스트 임베딩 선계산
    proj_dim = getattr(vt, "proj_out_dim", 1024)
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

    # 점수 계산: 이미지1 × 텍스트 배치 반복
    all_scores = torch.empty((1, M), dtype=torch.float32, device=device)
    t0 = 0
    while t0 < M:
        t1 = min(t0 + args.text_batch, M)
        cur_items = [{"text": texts[i], "label": 0} for i in range(t0, t1)]
        batch = collate(cur_items)
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)
        with torch.no_grad():
            with autocast_ctx:
                scores = forward_scores(vt, imgs_tensor, input_ids, attention_mask)  # [1, cur]
        all_scores[:, t0:t1] = scores
        t0 = t1

    # Top-K (라벨 중복 제거)
    scores_1d = all_scores[0].detach().cpu().to(torch.float32).numpy()
    chosen_idx, weights = topk_unique_by_label(
        scores_1d=scores_1d,
        labels_of_texts=labels_of_texts,
        k=args.topk,
        min_score=0.0,
    )

    # 결과 JSON 구성
    result: Dict[str, Any] = {
        "image": os.path.abspath(valid_paths[0]),
        "topk": [
            {
                "rank": r + 1,
                "text_index": int(ti),
                "label": int(labels_of_texts[ti]),
                "score": float(weights[r]),
                "text": texts[ti],
            }
            for r, ti in enumerate(chosen_idx)
        ],
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Demo] Saved Top-K JSON → {args.out_json}")

if __name__ == "__main__":
    main()
