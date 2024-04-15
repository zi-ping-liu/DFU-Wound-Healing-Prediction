"""
Stratified analysis on model performance

Author: Ziping Liu
Date: Apr 15, 2024
"""



# Import libraries
import pandas as pd
pd.set_option('display.precision', 3)
import numpy as np
import torch
from utils.clf_metrics import acc, sen, spe



def stratify_on_ulcer_size(df_in, train_csv_in, country, best_guids, mode = 'pat'):
    
    df = df_in.copy()
    train_csv = train_csv_in.copy()
    
    if country == 'US':
        train_csv = train_csv[train_csv['USorUK'] == 'US'].reset_index(drop = True)
        name_ = 'US'
    elif country == 'EU':
        train_csv = train_csv[train_csv['USorUK'] == 'UK'].reset_index(drop = True)
        name_ = 'EU'
    else:
        name_ = 'US + EU'
        
    # for q in np.linspace(0, 1, 11):
    #     thres = train_csv[['subject_number', 'cm2_planar_area']].drop_duplicates()['cm2_planar_area'].quantile(q)
    #     print(thres)
    thres_list = [0.08, 0.52, 1.32, 2.74, 5.96, 65.80, 10000]
    
    categs = [[] for _ in range(len(thres_list) - 1)]
    tags = [[] for _ in range(len(thres_list) - 1)]
    for idx in range(len(thres_list) - 1):
        if (idx == len(thres_list) - 2):
            lb, ub = 0, thres_list[idx + 1]
        else:
            lb, ub = thres_list[idx], thres_list[idx + 1]
        categs[idx] = train_csv[(train_csv['cm2_planar_area'] >= lb) & (train_csv['cm2_planar_area'] < ub)]['ICGUID'].tolist()
        tags[idx] = f"{name_} [{lb}, {ub})"
    
    res = pd.DataFrame()
    for ind, cur_guids in enumerate(categs):
        
        cur_df = df[df['ICGUID'].isin(cur_guids)].reset_index(drop = True)
        
        if mode == 'pat':
            num = len(cur_df[['subject_number', 'Visit Number']].drop_duplicates())
            pos = cur_df[['subject_number', 'Visit Number', 'GT']].drop_duplicates()['GT'].sum()
        elif mode == 'img':
            num = len(cur_df)
            pos = cur_df['GT'].sum()
        else:
            cur_df = cur_df[cur_df['ICGUID'].isin(best_guids)].reset_index(drop = True)
            num = len(cur_df)
            pos = cur_df['GT'].sum()
            
        tag = tags[ind]
        tag += f" - N = {num} || {pos} pos"
        
        if mode == 'pat': 
            preds = torch.tensor(cur_df.groupby(['subject_number', 'Visit Number'])['Pred_Proba'].mean().values)
            targets = torch.tensor(cur_df.groupby(['subject_number', 'Visit Number'])['GT'].mean().values)
        else:
            preds = torch.tensor(cur_df['Pred_Proba'].values)
            targets = torch.tensor(cur_df['GT'].values)

        info = {
            'Category': tag,
            'acc': None,
            'sen': None,
            'spe': None
        }
        for metric_fn in [acc, sen, spe]:
            try:
                val, _ = metric_fn(preds, targets, thres = 0.5)
            except:
                val = np.nan
            info[metric_fn.__name__] = val
            
        res = pd.concat([res, pd.DataFrame([info])], axis = 0).reset_index(drop = True)
            
    return res



def stratify_on_ulcer_loc(df_in, train_csv_in, country, best_guids, edge_cases, mode = 'pat'):
    
    df = df_in.copy()
    train_csv = train_csv_in.copy()
    
    if country == 'US':
        train_csv = train_csv[train_csv['USorUK'] == 'US'].reset_index(drop = True)
        name_ = 'US'
    elif country == 'EU':
        train_csv = train_csv[train_csv['USorUK'] == 'UK'].reset_index(drop = True)
        name_ = 'EU'
    else:
        name_ = 'US + EU'
    
    categs = [[] for _ in range(7)]
    # (a) Good-quality data
    categs[0] = train_csv[train_csv['DS_split'] != 'bad_quality']['ICGUID'].tolist()
    # (b) Good-quality non-toe data
    categs[1] = train_csv[(train_csv['DS_split'] != 'bad_quality')
                          & (train_csv['dfu_position'] != 'Toe')
                          ]['ICGUID'].tolist()
    # (c) Good-quality non-edge-case non-toe data
    categs[2] = train_csv[(train_csv['DS_split'] != 'bad_quality')
                          & (~train_csv['subject_number'].isin(edge_cases))
                          & (train_csv['dfu_position'] != 'Toe')
                          ]['ICGUID'].tolist()
    # (d) Good-quality edge-case non-toe data
    categs[3] = train_csv[(train_csv['DS_split'] != 'bad_quality')
                          & (train_csv['subject_number'].isin(edge_cases))
                          & (train_csv['dfu_position'] != 'Toe')
                          ]['ICGUID'].tolist()
    # (e) Good-quality toe data
    categs[4] = train_csv[(train_csv['DS_split'] != 'bad_quality')
                          & (train_csv['dfu_position'] == 'Toe')
                          ]['ICGUID'].tolist()
    # (f) Good-quality non-edge-case toe data
    categs[5] = train_csv[(train_csv['DS_split'] != 'bad_quality')
                          & (~train_csv['subject_number'].isin(edge_cases))
                          & (train_csv['dfu_position'] == 'Toe')
                          ]['ICGUID'].tolist()
    # (g) Good-quality edge-case toe data
    categs[6] = train_csv[(train_csv['DS_split'] != 'bad_quality')
                          & (train_csv['subject_number'].isin(edge_cases))
                          & (train_csv['dfu_position'] == 'Toe')
                          ]['ICGUID'].tolist()
    
    res = pd.DataFrame()
    for ind, cur_guids in enumerate(categs):
        
        cur_df = df[df['ICGUID'].isin(cur_guids)].reset_index(drop = True)
        
        if mode == 'pat':
            num = len(cur_df[['subject_number', 'Visit Number']].drop_duplicates())
            pos = cur_df[['subject_number', 'Visit Number', 'GT']].drop_duplicates()['GT'].sum()
        elif mode == 'img':
            num = len(cur_df)
            pos = cur_df['GT'].sum()
        else:
            cur_df = cur_df[cur_df['ICGUID'].isin(best_guids)].reset_index(drop = True)
            num = len(cur_df)
            pos = cur_df['GT'].sum()
        
        if ind == 0: categ = f"{name_} (good)"
        elif ind == 1: categ = f"{name_} non-toe (good)"
        elif ind == 2: categ = f"{name_} non-toe (good, non-edge)"
        elif ind == 3: categ = f"{name_} non-toe (good, edge)"
        elif ind == 4: categ = f"{name_} toe (good)"
        elif ind == 5: categ = f"{name_} toe (good, non-edge)"
        elif ind == 6: categ = f"{name_} toe (good, edge)"
        else: pass
        categ += f" - N = {num} || {pos} pos"
        
        if mode == 'pat':
            preds = torch.tensor(cur_df.groupby(['subject_number', 'Visit Number'])['Pred_Proba'].mean().values)
            targets = torch.tensor(cur_df.groupby(['subject_number', 'Visit Number'])['GT'].mean().values)
        else:
            preds = torch.tensor(cur_df['Pred_Proba'].values)
            targets = torch.tensor(cur_df['GT'].values)

        info = {
            'Category': categ,
            'acc': None,
            'sen': None,
            'spe': None
        }
        for metric_fn in [acc, sen, spe]:
            try:
                val, _ = metric_fn(preds, targets, thres = 0.5)
            except:
                val = np.nan
            info[metric_fn.__name__] = val
            
        res = pd.concat([res, pd.DataFrame([info])], axis = 0).reset_index(drop = True)
            
    return res



if __name__ == '__main__':
    
    train_csv = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404/src/data/WAUSI_unifiedv3_BSVp1-7_final1_20240411.csv")
    train_csv = train_csv[train_csv['good_ori'] == 'Y'].reset_index(drop = True)
    train_csv = train_csv[~train_csv['DS_split'].isin(['bad_quality', 'exclude_from_classification'])].reset_index(drop = True)
    
    # Subjects whose clinical features remain to be verified
    # subjects_to_drop = ['201-018', '202-042', '202-047', '202-048', '202-050', '202-052', '202-056', '202-062', '202-067', '202-078', '202-082', '203-091', '292-044', '292-045', '292-024']
    # train_csv = train_csv[~train_csv['subject_number'].isin(subjects_to_drop)].reset_index(drop = True)
    
    train_csv = train_csv[train_csv['USorUK'] == 'US'].reset_index(drop = True)
    
    prediction_path = "/home/efs/ziping/workspaces/dfu/clf_algo_release_202404/results/240407/baseline"
    hs = 1526
    print(f"hs = {hs}")
    
    # df_pred = pd.read_csv(f"{prediction_path}/hs_{hs}/predictions_cv.csv")
    df_pred = pd.read_csv(f"{prediction_path}/hs_{hs}/predictions_test.csv")
    df_pred = df_pred[df_pred['Visit Number'] == 'DFU_SV1'].reset_index(drop = True)
    
    # Gather ImgCollGUIDs that yield best orientation for each patient
    train_csv['orientation_deg'].fillna(float('inf'), inplace = True)
    best_idx = train_csv.groupby(['subject_number', 'Visit Number'])['orientation_deg'].idxmin().values
    best_guids = train_csv.iloc[best_idx]['ICGUID'].values.tolist()
        
    strat_area_us_pat = stratify_on_ulcer_size(df_pred, train_csv, 'US', best_guids, mode = 'pat')
    strat_area_us_img = stratify_on_ulcer_size(df_pred, train_csv, 'US', best_guids, mode = 'img')
    strat_area_us_best = stratify_on_ulcer_size(df_pred, train_csv, 'US', best_guids, mode = 'best')
    print(strat_area_us_pat)
    print(strat_area_us_img)
    print(strat_area_us_best)
    print("")
        
    # strat_loc_us_pat = stratify_on_ulcer_loc(df_pred, train_csv, 'US', best_guids, edge_cases, mode = 'pat')
    # strat_loc_us_img = stratify_on_ulcer_loc(df_pred, train_csv, 'US', best_guids, edge_cases, mode = 'img')
    # strat_loc_us_best = stratify_on_ulcer_loc(df_pred, train_csv, 'US', best_guids, edge_cases, mode = 'best')