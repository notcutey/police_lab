import os
import numpy as np
from tqdm import tqdm

import torch
from time import strftime, localtime

import pandas as pd
import ast
import re

from loguru import logger


# ============================
#  공통 유틸 함수들
# ============================

def safe_literal_eval(text):
    """
    문자열에서 리스트 형태([ ... ])만 추출한 뒤 literal_eval로 안전하게 파싱.
    실패 시 빈 리스트 반환.
    """
    try:
        clean = re.findall(r"\[[^\[\]]+\]", text)
        if clean:
            return ast.literal_eval(clean[0])
    except Exception as e:
        print("⚠️ ast.literal_eval 실패:", text)
    return []


def compute_average_precision(top_ranks, gt_indices, max_rank=None):
    """
    AP(Average Precision) 계산 함수. (기존 그대로 유지)
    """
    # tensor -> numpy
    if isinstance(top_ranks, torch.Tensor):
        top_ranks = top_ranks.detach().cpu().numpy()
    else:
        top_ranks = np.asarray(top_ranks)

    gt_set = set(gt_indices)
    if len(gt_set) == 0:
        return 0.0

    # 상위 max_rank까지만 보고 싶으면 자르기
    if max_rank is not None:
        top_ranks = top_ranks[:max_rank]

    # 각 순위에서 정답 여부 (True/False)
    relevant = np.array([idx in gt_set for idx in top_ranks], dtype=bool)

    if not relevant.any():
        # 랭킹 안에 정답이 하나도 없을 때
        return 0.0

    precisions = []
    rel_so_far = 0
    for i, is_rel in enumerate(relevant):
        if is_rel:
            rel_so_far += 1
            # i는 0부터 시작 → 순위는 i+1
            precisions.append(rel_so_far / float(i + 1))

    # 분모는 "전체 정답 개수"로 두는 전형적인 AP 정의
    ap = float(np.sum(precisions)) / float(len(gt_set))
    return ap


# ============================
#  nDCG 계산 함수
# ============================

def compute_ndcg(top_ranks, gt_indices, max_rank=None):
    """
    nDCG (Normalized Discounted Cumulative Gain) 계산 함수.
    여기서는 binary relevance (정답이면 1, 아니면 0) 기준으로 계산.

    DCG@K = sum_i ( rel_i / log2(rank_i + 1) ),  rank_i = i+1 (1-based)
    IDCG@K = 정답들을 가장 앞에 모았을 때의 DCG@K
    nDCG = DCG / IDCG
    """
    # tensor -> numpy
    if isinstance(top_ranks, torch.Tensor):
        top_ranks = top_ranks.detach().cpu().numpy()
    else:
        top_ranks = np.asarray(top_ranks)

    gt_set = set(gt_indices)
    if len(gt_set) == 0:
        return 0.0

    # 상위 max_rank까지만 보고 싶으면 자르기
    if max_rank is not None:
        top_ranks = top_ranks[:max_rank]

    # 각 순위에서 정답 여부 (True/False)
    relevant = np.array([idx in gt_set for idx in top_ranks], dtype=bool)
    num_rel = relevant.sum()
    if num_rel == 0:
        return 0.0

    # ----- DCG -----
    # i: 0-based index, rank = i+1
    dcg = 0.0
    for i, is_rel in enumerate(relevant):
        if is_rel:
            rank = i + 1
            dcg += 1.0 / np.log2(rank + 1.0)  # log2(rank+1): rank=1→log2(2)=1

    # ----- IDCG -----
    # 이상적인 경우: 정답이 맨 앞에 모여있다고 가정
    k = len(top_ranks)
    max_hits = min(num_rel, k)
    idcg = 0.0
    for i in range(max_hits):
        rank = i + 1
        idcg += 1.0 / np.log(rank + 1.0)

    if idcg == 0.0:
        return 0.0

    ndcg = float(dcg / idcg)
    return ndcg


# ============================
#  Ensemble + nDCG 로그 함수
# ============================

def emm_log_print(sorted_indices, sorted_distances, query_order, ref_indices, args):
    """
    EMM(ensemble) 결과에 대한 mean nDCG 계산만 수행.

    Parameters
    ----------
    sorted_indices : tensor [num_queries, num_refs]
        각 쿼리에 대해 distance 오름차순으로 정렬된 ref 인덱스 (기본 retrieval 결과)
    sorted_distances : tensor [num_queries, num_refs]
        각 쿼리에 대해 distance 오름차순으로 정렬된 거리값
    query_order : list[str]
        쿼리 파일명 리스트 (df['cropped']와 매칭)
    ref_indices : dict[str, int]
        { ref_파일명(or label) : ref_index } 매핑
    args : argparse.Namespace
        args.testcsv, args.emmcsv 등 포함

    Returns
    -------
    mean_ndcg_percent : float
        전체 쿼리에 대한 mean nDCG (% 단위, 0~100)
    """
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    cur_fname = os.path.basename(__file__).rstrip('.py')
    log_save_dir = os.path.join('./logs', f'{cur_fname}_{start_time_stamp}.log')

    logger.add(log_save_dir, format="{message}", level="DEBUG")
    
    df = pd.read_csv(args.testcsv)
    emm_csv = pd.read_csv(args.emmcsv)

    # 쿼리별 nDCG 리스트
    ndcg_list = []

    for qidx, (top_ranks, top_distances, query_file_name) in tqdm(
        enumerate(zip(sorted_indices, sorted_distances, query_order))
    ):
        # GT 패턴들 (문양 label 리스트 문자열)
        gt_file_name = df[df['cropped'] == query_file_name]['gt'].iloc[0]
        gt_labels = ast.literal_eval(gt_file_name)
        gt_indices = [ref_indices[gt_label] for gt_label in gt_labels]

        # EMM에서 제공하는 상위 100 후보 (문양 label 리스트)
        emm_top100 = ast.literal_eval(
            emm_csv[emm_csv['cropped'] == query_file_name]['gt'].iloc[0]
        )
        emm_top100_indices = list(map(lambda x: ref_indices[x], emm_top100))
        
        # 문양이 100개 이상이 아닐 때는 기본 top_ranks에서 채워넣기
        if len(emm_top100_indices) < 100:
            for top_idx in top_ranks:
                if top_idx not in emm_top100_indices:
                    emm_top100_indices.append(int(top_idx))
                if len(emm_top100_indices) >= 100:
                    break
        
        emm_top100_indices = torch.tensor(
            emm_top100_indices,
            dtype=torch.long,
            device=top_ranks.device
        )

        # 각 후보(ref index)가 original ranking에서 몇 등인지 구함
        selected_values = top_ranks[emm_top100_indices]
        # 그 순위를 기준으로 다시 정렬 → ensemble ranking
        sorted_idx_within = torch.argsort(selected_values)
        ensemble_indices = emm_top100_indices[sorted_idx_within]   # 최종 rerank 결과 (ref index들의 순서)

        # ====== nDCG 계산 (여기서는 top-100 ranking 기준) ======
        ndcg = compute_ndcg(ensemble_indices, gt_indices, max_rank=None)
        ndcg_list.append(ndcg)
        # ========================================================

        logger.debug(f"[EMM] query: {query_file_name}, gt_labels: {gt_labels}, nDCG: {ndcg:.4f}")

    # 최종 mean nDCG 계산
    if len(ndcg_list) > 0:
        mean_ndcg_percent = float(np.mean(ndcg_list)) * 100.0  # %
    else:
        mean_ndcg_percent = 0.0

    logger.debug(f'FINAL EMM mean nDCG: {mean_ndcg_percent:.4f} %')

    return mean_ndcg_percent


# ============================
#  기본 retrieval + nDCG 로그 함수
# ============================

def log_print(sorted_indices, sorted_distances, query_order, ref_indices, ref_names=None, args=None):
    """
    기본 retrieval 결과에 대한 mean nDCG 계산만 수행.
    """
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    cur_fname = os.path.basename(__file__).rstrip('.py')
    log_save_dir = os.path.join('./logs', f'{cur_fname}_{start_time_stamp}.log')

    logger.add(log_save_dir, format="{message}", level="DEBUG")
    
    # GT CSV (경로는 필요에 맞게 바꾸셔도 됨)
    if args is not None and hasattr(args, "testcsv"):
        gt_csv_path = args.testcsv
    else:
        # 기존 경로 유지
        gt_csv_path = "/root/project/llm_prompt/label_test_multimodal.csv"

    df = pd.read_csv(gt_csv_path)

    # 쿼리별 nDCG 리스트
    ndcg_list = []
    skipped = 0

    for qidx, (top_ranks, top_distances, query_file_name) in tqdm(
        enumerate(zip(sorted_indices, sorted_distances, query_order))
    ):
        try:
            gt_file_name = df[df['cropped'] == query_file_name]['gt'].iloc[0]
        except Exception:
            skipped += 1
            continue

        # 문자열 "[...]" -> 리스트
        gt_labels = ast.literal_eval(gt_file_name)
        gt_indices = [ref_indices[gt_label] for gt_label in gt_labels]

        # ====== nDCG 계산 (full ranking 기준) ======
        ndcg = compute_ndcg(top_ranks, gt_indices, max_rank=None)
        ndcg_list.append(ndcg)
        # ===========================================

        logger.debug(f"[BASE] query: {query_file_name}, gt_labels: {gt_labels}, nDCG: {ndcg:.4f}")

    # 최종 mean nDCG
    if len(ndcg_list) > 0:
        mean_ndcg_percent = float(np.mean(ndcg_list)) * 100.0  # %
    else:
        mean_ndcg_percent = 0.0

    logger.debug(f"Skipped queries (no GT found): {skipped}")
    logger.debug(f'FINAL BASE mean nDCG: {mean_ndcg_percent:.4f} %')

    return mean_ndcg_percent
