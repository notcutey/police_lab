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
    AP(Average Precision) 계산 함수.

    Parameters
    ----------
    top_ranks : 1D tensor or array-like
        검색 결과로 나온 ref 인덱스들이 랭킹 순서대로 나열된 것
        (예: 길이 = 전체 ref 개수)
    gt_indices : list[int]
        이 쿼리의 정답(ref 인덱스들, 여러 개 가능)
    max_rank : int or None, optional
        상위 몇 개까지만 평가에 사용할지 지정.
        None이면 전체 top_ranks 사용.

    Returns
    -------
    ap : float
        0.0 ~ 1.0 사이의 Average Precision 값
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
#  Ensemble + mAP 로그 함수
# ============================

def emm_log_print(sorted_indices, sorted_distances, query_order, ref_indices, args):
    """
    EMM(ensemble) 결과에 대한 mAP 계산만 수행.

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
    mAP_percent : float
        전체 쿼리에 대한 mAP (% 단위, 0~100)
    """
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    cur_fname = os.path.basename(__file__).rstrip('.py')
    log_save_dir = os.path.join('./logs', f'{cur_fname}_{start_time_stamp}.log')

    logger.add(log_save_dir, format="{message}", level="DEBUG")
    
    df = pd.read_csv(args.testcsv)
    emm_csv = pd.read_csv(args.emmcsv)

    # 쿼리별 AP 리스트
    ap_list = []

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

        # ====== mAP용 AP 계산 (여기서는 top-100 ranking 기준) ======
        ap = compute_average_precision(ensemble_indices, gt_indices, max_rank=None)
        ap_list.append(ap)
        # ========================================================

        logger.debug(f"[EMM] query: {query_file_name}, gt_labels: {gt_labels}, AP: {ap:.4f}")

    # 최종 mAP 계산
    if len(ap_list) > 0:
        mAP_percent = float(np.mean(ap_list)) * 100.0  # %
    else:
        mAP_percent = 0.0

    logger.debug(f'FINAL EMM mAP: {mAP_percent:.4f} %')

    return mAP_percent


# ============================
#  기본 retrieval + mAP 로그 함수
# ============================

def log_print(sorted_indices, sorted_distances, query_order, ref_indices, ref_names=None, args=None):
    """
    기본 retrieval 결과에 대한 mAP 계산만 수행.

    Parameters
    ----------
    sorted_indices : tensor [num_queries, num_refs]
        각 쿼리에 대해 distance 오름차순으로 정렬된 ref 인덱스
    sorted_distances : tensor [num_queries, num_refs]
        (현재 mAP 계산에는 사용하지 않지만, 인터페이스 유지용)
    query_order : list[str]
        쿼리 파일명 리스트 (df['cropped']와 매칭)
    ref_indices : dict[str, int]
        { ref_파일명(or label) : ref_index } 매핑
    ref_names : list[str] or None
        각 ref index에 대응되는 이름(파일명) 리스트 (옵션, 여기선 사용 X)
    args : argparse.Namespace or None
        필요하면 경로 등을 여기에 넣어 사용

    Returns
    -------
    mAP_percent : float
        전체 쿼리에 대한 mAP (% 단위, 0~100)
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

    # 쿼리별 AP 리스트
    ap_list = []
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

        # ====== mAP용 AP 계산 (full ranking 기준) ======
        ap = compute_average_precision(top_ranks, gt_indices, max_rank=None)
        ap_list.append(ap)
        # ===========================================

        logger.debug(f"[BASE] query: {query_file_name}, gt_labels: {gt_labels}, AP: {ap:.4f}")

    # 최종 mAP
    if len(ap_list) > 0:
        mAP_percent = float(np.mean(ap_list)) * 100.0  # %
    else:
        mAP_percent = 0.0

    logger.debug(f"Skipped queries (no GT found): {skipped}")
    logger.debug(f'FINAL BASE mAP: {mAP_percent:.4f} %')

    return mAP_percent
