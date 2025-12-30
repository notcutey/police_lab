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
import re
import shutil
import torch

from loguru import logger


def emm_log_print(sorted_indices, sorted_distances, query_order, ref_indices, args):
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    cur_fname = os.path.basename(__file__).rstrip('.py')
    log_save_dir = os.path.join('./logs', f'{cur_fname}_{start_time_stamp}.log')

    logger.add(log_save_dir, format="{message}", level="DEBUG")
    
    output_shape = []
    
    df = pd.read_csv(args.testcsv)
    emm_csv = pd.read_csv(args.emmcsv)
    gt_len = sorted_distances.shape[1]

    count_1pct, count_5pct, count_10pct, count_50, total = [0] * 5

    print(query_order)

    ONE_PCNT = int(gt_len * 0.01) # top 1
    FIVE_PCNT = int(gt_len * 0.05) # top 5
    TEN_PCNT = int(gt_len * 0.1) # top 10
    TOP_50 = 20 # top_50

    for qidx, (top_ranks, top_distances, query_file_name) in tqdm(enumerate(zip(sorted_indices, sorted_distances, query_order))):
        gt_file_name = df[df['cropped'] == query_file_name]['gt'].iloc[0]  # 첫 번째 일치하는 값 가져오기
        gt_indices = [ref_indices[gt_label] for gt_label in ast.literal_eval(gt_file_name)]

        emm_top100 = ast.literal_eval(emm_csv[emm_csv['cropped'] == query_file_name]['gt'].iloc[0])
        emm_top100_indices = list(map(lambda x: ref_indices[x], emm_top100))
        
         # 문양이 50개 이상이 아닐 때도 있음
        if len(emm_top100_indices) < 100:
            for top_idx in top_ranks:
                if top_idx not in emm_top100_indices:
                    emm_top100_indices.append(top_idx)

                if len(emm_top100_indices) >= 100:
                    break
        
        emm_top100_indices = torch.tensor(emm_top100_indices)
        selected_values = top_ranks[emm_top100_indices]
        sorted_indices = torch.argsort(selected_values)
        ensemble_indices = emm_top100_indices[sorted_indices]
        
        # top-1%, top-50 개 추출
        top1pct_list = emm_top100_indices[:ONE_PCNT]
        top50_list = emm_top100_indices[:TOP_50]
        
        # top index에 있으면 카운팅
        flag1, flag2 = False, False
        for gt_index in gt_indices:
            emm_involve_index = (ensemble_indices == gt_index).nonzero(as_tuple=True)[0]
            if len(emm_involve_index) and emm_involve_index <= ONE_PCNT and not flag1:
                flag1 = True
                count_1pct += 1
            if  len(emm_involve_index) and emm_involve_index <= TOP_50 and not flag2:
                flag2 = True
                count_50 += 1
 
                
        total += 1

        # import ipdb; ipdb.set_trace()
        logger.debug(f"query_number: {query_file_name}, gt_query_label:{gt_file_name}")
        # logger.debug(f"top 1% distance index: {[i.item() for i in top1pct_list]}")
        # logger.debug(f"top 50 distance index: {top50_list}")
        # logger.debug(f'count top 1% accuracy: {count_1pct} / total: {total}')
        # logger.debug(f'count top 5% accuracy: {count_5pct} / total: {total}')
        # logger.debug(f'count top 10% accuracy: {count_10pct} / total: {total}')
        # logger.debug(f'count top 50 accuracy: {count_50} / total: {total}\n')
    
    logger.debug(f"query_feat.shape: {output_shape}")
    logger.debug(f'Final top 1% accuracy: {(count_1pct / total) * 100} %')
    logger.debug(f'FINAL top 5% accuracy: {(count_5pct / total) * 100} %')
    logger.debug(f'FINAL top 10% accuracy: {(count_10pct / total) * 100} %')
    logger.debug(f'FINAL top 50 accuracy: {(count_50 / total) * 100} %')

    return [(count_1pct / total) * 100, (count_5pct / total) * 100, (count_10pct / total) * 100, (count_50 / total) * 100]

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

def log_print(sorted_indices, sorted_distances, query_order, ref_indices, ref_names=None, args=None):
    start_time_stamp = strftime("%m-%d_%H%M", localtime())
    cur_fname = os.path.basename(__file__).rstrip('.py')
    log_save_dir = os.path.join('./logs', f'{cur_fname}_{start_time_stamp}.log')

    logger.add(log_save_dir, format="{message}", level="DEBUG")
    # logger.debug("This is a debug message")

    # logging.basicConfig(filename=log_save_dir, \
    #         level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    
    output_shape = []
    names = []
    lists = []

    # gt_loader = DataLoader(gt_dataset, batch_size=1, shuffle=False)
    df = pd.read_csv("/root/project/llm_prompt/label_3066.csv")
    gt_len = sorted_distances.shape[1]

    count_1pct, count_5pct, count_10pct, count_50, total = [0] * 5

    ONE_PCNT = int(gt_len * 0.01) # top 1
    FIVE_PCNT = int(gt_len * 0.05) # top 5
    TEN_PCNT = int(gt_len * 0.1) # top 10
    TOP_50 = 50 # top_50

    csv_file = 'PL_totalref_top50_results.csv'
    cnt = 0
    # with open(csv_file, 'w', newline='') as csvfile:
    # writer = csv.writer(csvfile)
    # writer.writerow(['filename', 'top_50_indices', 'top_50_distances'])
    for qidx, (top_ranks, top_distances, query_file_name) in tqdm(enumerate(zip(sorted_indices, sorted_distances, query_order))):
        try:
            gt_file_name = df[df['cropped'] == query_file_name]['gt'].iloc[0]  # 첫 번째 일치하는 값 가져오기
        except:
            cnt += 1
            continue
        # gt_file_name = safe_literal_eval(gt_file_name)
        print(gt_file_name)
        gt_indices = [ref_indices[gt_label] for gt_label in ast.literal_eval(gt_file_name)]
        

        top1pct_list = top_ranks[:ONE_PCNT]
        top5pct_list = top_ranks[:FIVE_PCNT]
        top10pct_list = top_ranks[:TEN_PCNT]
        top50_list = top_ranks[:TOP_50] # 0 부터
        
        top50_distances = top_distances[:TOP_50]
        # top index에 있으면 카운팅
        flag1, flag2, flag3, flag4 = False, False, False, False
        for gt_index in gt_indices:
            involve_index = (top_ranks == gt_index).nonzero(as_tuple=True)[0]
            if involve_index <= ONE_PCNT and not flag1:
                flag1 = True
                count_1pct += 1
            if involve_index <= FIVE_PCNT and not flag2:
                flag2 = True
                count_5pct += 1
            if involve_index <= TEN_PCNT and not flag3:
                flag3 = True
                count_10pct += 1
            if involve_index <= TOP_50 and not flag4:
                flag4 = True
                count_50 += 1
                
                sample_save = False
                if sample_save:
                    import os.path as osp
                    from_path = './data/PoliceLab/도메인 분리/transper_paper'
                    to_path = './data/PoliceLab/sample'
                    shutil.copyfile(osp.join(from_path, query_file_name) + '.png', osp.join(to_path, query_file_name) + '.png')
                
        total += 1

        # import ipdb; ipdb.set_trace()
        logger.debug(f"query_number: {query_file_name}, gt_query_label:{gt_file_name}")
        # logger.debug(f"top 1% distance index: {[i.item() for i in top1pct_list]}")
        # logger.debug(f"top 5% distance index: {[i.item() for i in top5pct_list]}")
        # logger.debug(f"top 10% distance index: {[i.item() for i in top10pct_list]}")
        logger.debug(f"top 50 distance index: {top50_list}")
        logger.debug(f'count top 1% accuracy: {count_1pct} / total: {total}')
        logger.debug(f'count top 5% accuracy: {count_5pct} / total: {total}')
        logger.debug(f'count top 10% accuracy: {count_10pct} / total: {total}')
        logger.debug(f'count top 50 accuracy: {count_50} / total: {total}\n')

        # values = top50_distances.cpu().tolist()

        # min_val = values[0] - 0.05
        # max_val = values[-1] + 0.01

        # # 백분위수 계산
        # similarity = [(value - min_val) / (max_val - min_val) * 100 for value in values][::-1]

        # top50_filenames = list(map(lambda x: list(ref_indices.keys())[x], top50_list.cpu().tolist()))

        # writer.writerow([query_file_name] + [top50_filenames] + [similarity])

        # # if qidx % 25 == 0:
        # #     del gt_indices
        # #     gc.collect()
        # #     torch.cuda.empty_cache()
        # tmp = []
        # names.append(query_file_name)

        # for j in top_ranks:
        #     tmp.append(ref_names[j])

        # lists.append(tmp[:])
    
    logger.debug(f"query_feat.shape: {output_shape}")
    logger.debug(f'Final top 1% accuracy: {(count_1pct / total) * 100} %')
    logger.debug(f'FINAL top 5% accuracy: {(count_5pct / total) * 100} %')
    logger.debug(f'FINAL top 10% accuracy: {(count_10pct / total) * 100} %')
    logger.debug(f'FINAL top 50 accuracy: {(count_50 / total) * 100} %')

    # df = pd.DataFrame({'cropped': names, 'gt': lists})
    # df.to_csv('retrieval_results_top100.csv', index=False)

    print('불발', cnt)
    return [(count_1pct / total) * 100, (count_5pct / total) * 100, (count_10pct / total) * 100, (count_50 / total) * 100]
