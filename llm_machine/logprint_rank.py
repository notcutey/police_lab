import os
import numpy as np
from tqdm import tqdm

import logging
from time import strftime, localtime

import pandas as pd
import ast


def get_logger(log_dir: str = "./logs", name: str = "log_print"):
    """
    간단한 파일 로거 생성
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        time_stamp = strftime("%m-%d_%H-%M", localtime())
        log_path = os.path.join(log_dir, f"log_{time_stamp}.log")

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger


def log_print(sorted_indices,
              sorted_distances,
              query_order,
              ref_indices,
              ref_names,
              args):
    """
    Args:
        sorted_indices: torch.Tensor [num_query, num_ref]
        sorted_distances: torch.Tensor [num_query, num_ref]
        query_order: List[str] (확장자 제거된 query 파일명)
        ref_indices: Dict[str -> int]
        ref_names: List[str] (index -> ref 이름)  (사용 X)
        args: argparse.Namespace
    """

    logger = get_logger(log_dir="./logs", name="log_print")

    # Top-k 설정  
    TOP_50 = 50
    TOP_500 = 500
    TOP_1000 = 1000
    TOP_2000 = 2000
    TOP_3000 = 3000

    count_50 = 0
    count_500 = 0
    count_1000 = 0
    count_2000 = 0
    count_3000 = 0
    total = 0

    # CSV 저장용
    rank_rows = []

    # GT CSV 로드
    df = pd.read_csv("/root/project/llm_prompt/label_3066.csv")

    logger.debug(f"sorted_indices.shape: {sorted_indices.shape}")
    logger.debug(f"sorted_distances.shape: {sorted_distances.shape}")

    # Query loop
    for qidx, (top_ranks, top_distances, query_file_name) in tqdm(
        enumerate(zip(sorted_indices, sorted_distances, query_order)),
        total=len(query_order)
    ):
        # GT 찾기
        try:
            gt_file_name = df[df["cropped"] == query_file_name]["gt"].iloc[0]
        except IndexError:
            logger.debug(f"[WARN] GT not found for query: {query_file_name}")
            rank_rows.append({
                "test_name": query_file_name,
                "score": -1
            })
            continue

        # GT 라벨 파싱
        try:
            gt_labels = ast.literal_eval(gt_file_name)
            if isinstance(gt_labels, str):
                gt_labels = [gt_labels]
        except Exception:
            gt_labels = [gt_file_name]

        # GT ref index로 변환
        gt_indices = []
        for gt_label in gt_labels:
            if gt_label in ref_indices:
                gt_indices.append(ref_indices[gt_label])
            else:
                logger.debug(
                    f"[WARN] GT label '{gt_label}' not in ref_indices for query '{query_file_name}'"
                )

        # --------- 전체 ref 순위 기준 best_rank 계산 ---------
# --------- 전체 ref 순위 기준 best_rank 계산 ---------
        best_rank = float("inf")

        if len(gt_indices) > 0:
            for gt_index in gt_indices:
                involve_index = (top_ranks == gt_index).nonzero(as_tuple=True)[0]
                if len(involve_index) > 0:
                    rank0 = involve_index[0].item()   # 0-based
                    rank1 = rank0 + 1                # 1-based 로 변경
                    if rank1 < best_rank:
                        best_rank = rank1
        # ------------------------------------------------------

        # top-k accuracy (1-based 기준)
        if best_rank != float("inf"):
            if best_rank <= TOP_50:
                count_50 += 1
            if best_rank <= TOP_500:
                count_500 += 1
            if best_rank <= TOP_1000:
                count_1000 += 1
            if best_rank <= TOP_2000:
                count_2000 += 1
            if best_rank <= TOP_3000:
                count_3000 += 1

        total += 1

        # CSV score도 그대로 best_rank 사용 (이미 1-based)
        score = -1 if best_rank == float("inf") else best_rank

        rank_rows.append({
            "test_name": query_file_name,
            "score": score
})


        logger.debug(f"query: {query_file_name}, best_rank: {best_rank}")

    # 최종 accuracy 로그
    if total > 0:
        logger.debug(f"Final top 50 accuracy: {(count_50 / total) * 100} %")
        logger.debug(f"FINAL top 500 accuracy: {(count_500 / total) * 100} %")
        logger.debug(f"FINAL top 1000 accuracy: {(count_1000 / total) * 100} %")
        logger.debug(f"FINAL top 2000 accuracy: {(count_2000 / total) * 100} %")
        logger.debug(f"FINAL top 3000 accuracy: {(count_3000 / total) * 100} %")

    # -------- CSV 저장 --------
    csv_path = "./rank_result.csv"
    pd.DataFrame(rank_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved GT rank CSV to {csv_path}")
    # ---------------------------

    if total == 0:
        return [0.0] * 5

    return [
        (count_50 / total) * 100,
        (count_500 / total) * 100,
        (count_1000 / total) * 100,
        (count_2000 / total) * 100,
        (count_3000 / total) * 100,
    ]
