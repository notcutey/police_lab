import os
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F

import logging
from time import strftime, localtime

import ipdb
import pandas as pd

import csv
import ast

import shutil
import torch

from loguru import logger


# ============================
#  mAP 계산 함수
# ============================
def compute_average_precision(top_ranks, gt_indices, max_rank=None):
    """
    AP(Average Precision) 계산 함수.
    """
    if isinstance(top_ranks, torch.Tensor):
        top_ranks = top_ranks.detach().cpu().numpy()
    else:
        top_ranks = np.asarray(top_ranks)

    gt_set = set(gt_indices)
    if len(gt_set) == 0:
        return 0.0

    if max_rank is not None:
        top_ranks = top_ranks[:max_rank]

    relevant = np.array([idx in gt_set for idx in top_ranks], dtype=bool)

    if not relevant.any():
        return 0.0

    precisions = []
    rel_so_far = 0
    for i, is_rel in enumerate(relevant):
        if is_rel:
            rel_so_far += 1
            precisions.append(rel_so_far / float(i + 1))

    ap = float(np.sum(precisions)) / float(len(gt_set))
    return ap


# ============================
#  nDCG 계산 함수
# ============================
def compute_ndcg(top_ranks, gt_indices, max_rank=None):
    """
    nDCG (Normalized Discounted Cumulative Gain) 계산 함수.
    binary relevance (정답이면 1, 아니면 0) 기준.
    """
    if isinstance(top_ranks, torch.Tensor):
        top_ranks = top_ranks.detach().cpu().numpy()
    else:
        top_ranks = np.asarray(top_ranks)

    gt_set = set(gt_indices)
    if len(gt_set) == 0:
        return 0.0

    if max_rank is not None:
        top_ranks = top_ranks[:max_rank]

    relevant = np.array([idx in gt_set for idx in top_ranks], dtype=bool)
    num_rel = relevant.sum()
    if num_rel == 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, is_rel in enumerate(relevant):
        if is_rel:
            rank = i + 1
            dcg += 1.0 / np.log2(rank + 1.0)

    # IDCG (이상적인 경우: 정답이 맨 앞에 모여있음)
    k = len(top_ranks)
    max_hits = min(num_rel, k)
    idcg = 0.0
    for i in range(max_hits):
        rank = i + 1
        idcg += 1.0 / np.log2(rank + 1.0)

    if idcg == 0.0:
        return 0.0

    ndcg = float(dcg / idcg)
    return ndcg


# ============================
#  메인 log_print 함수
# ============================
def log_print(sorted_indices, sorted_distances, query_order, ref_indices, ref_names=None, args=None):
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    cur_fname = os.path.basename(__file__).rstrip('.py')
    log_save_dir = os.path.join('./logs', f'{cur_fname}_{start_time_stamp}.log')

    logger.add(log_save_dir, format="{message}", level="DEBUG")

    output_shape = []
    names = []
    lists = []

    df = pd.read_csv(args.testcsv)
    gt_len = sorted_distances.shape[1]

    # 전체 reference 개수
    total_refs = sorted_distances.shape[1]

    count_50, count_1pct, count_5pct, count_10pct, total = [0] * 5

    TOP_50 = 50
    TOP_1PCT = int(np.floor(total_refs * 0.01))
    TOP_5PCT = int(np.floor(total_refs * 0.05))
    TOP_10PCT = int(np.floor(total_refs * 0.10))

    print(f"Total references: {total_refs}")
    print(f"TOP 50: {TOP_50}")
    print(f"TOP 1%: {TOP_1PCT}")
    print(f"TOP 5%: {TOP_5PCT}")
    print(f"TOP 10%: {TOP_10PCT}")

    # mAP, nDCG 리스트 추가
    ap_list = []
    ndcg_list = []

    for qidx, (top_ranks, top_distances, query_file_name) in tqdm(enumerate(zip(sorted_indices, sorted_distances, query_order))):
        try:
            gt_file_name = df[df['cropped'] == query_file_name]['gt'].iloc[0]
        except:
            continue
        gt_indices_list = [ref_indices[gt_label] for gt_label in ast.literal_eval(gt_file_name)]

        top50_list = top_ranks[:TOP_50]
        top1pct_list = top_ranks[:TOP_1PCT]
        top5pct_list = top_ranks[:TOP_5PCT]
        top10pct_list = top_ranks[:TOP_10PCT]

        top50_distances = top_distances[:TOP_50]

        # Top-K accuracy 계산
        best_rank = float('inf')

        for gt_index in gt_indices_list:
            involve_index = (top_ranks == gt_index).nonzero(as_tuple=True)[0]
            if len(involve_index) > 0:
                rank = involve_index[0].item()
                if rank < best_rank:
                    best_rank = rank

        if best_rank != float('inf'):
            if best_rank < TOP_50:
                count_50 += 1
            if best_rank < TOP_1PCT:
                count_1pct += 1
            if best_rank < TOP_5PCT:
                count_5pct += 1
            if best_rank < TOP_10PCT:
                count_10pct += 1

                sample_save = False
                if sample_save:
                    import os.path as osp
                    from_path = './data/PoliceLab/도메인 분리/transper_paper'
                    to_path = './data/PoliceLab/sample'
                    shutil.copyfile(osp.join(from_path, query_file_name) + '.png', osp.join(to_path, query_file_name) + '.png')

        total += 1

        # mAP 계산
        ap = compute_average_precision(top_ranks, gt_indices_list, max_rank=None)
        ap_list.append(ap)

        # nDCG 계산
        ndcg = compute_ndcg(top_ranks, gt_indices_list, max_rank=None)
        ndcg_list.append(ndcg)

        logger.debug(f"query_number: {query_file_name}, gt_query_label:{gt_file_name}")
        logger.debug(f"top 50 distance index: {top50_list}")
        logger.debug(f'count top 50 accuracy: {count_50} / total: {total}')
        logger.debug(f'count top 1% accuracy: {count_1pct} / total: {total}')
        logger.debug(f'count top 5% accuracy: {count_5pct} / total: {total}')
        logger.debug(f'count top 10% accuracy: {count_10pct} / total: {total}')
        logger.debug(f'AP: {ap:.4f}, nDCG: {ndcg:.4f}\n')

    # 최종 mAP, nDCG 계산
    if len(ap_list) > 0:
        mAP_percent = float(np.mean(ap_list)) * 100.0
    else:
        mAP_percent = 0.0

    if len(ndcg_list) > 0:
        mean_ndcg_percent = float(np.mean(ndcg_list)) * 100.0
    else:
        mean_ndcg_percent = 0.0

    logger.debug(f"query_feat.shape: {output_shape}")
    logger.debug(f'Final top 50 accuracy: {(count_50 / total) * 100:.4f} %')
    logger.debug(f'FINAL top 1% accuracy: {(count_1pct / total) * 100:.4f} %')
    logger.debug(f'FINAL top 5% accuracy: {(count_5pct / total) * 100:.4f} %')
    logger.debug(f'FINAL top 10% accuracy: {(count_10pct / total) * 100:.4f} %')
    logger.debug(f'FINAL mAP: {mAP_percent:.4f} %')
    logger.debug(f'FINAL nDCG: {mean_ndcg_percent:.4f} %')

    return [
        (count_50 / total) * 100,
        (count_1pct / total) * 100,
        (count_5pct / total) * 100,
        (count_10pct / total) * 100,
        mAP_percent,
        mean_ndcg_percent
    ]
