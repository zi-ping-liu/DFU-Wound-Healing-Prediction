"""
Perform any necessary changes to unified csv

Author: Ziping Liu
Date: Apr 7, 2024
"""



# Import libraries
import pandas as pd

df = pd.read_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404/src/data/WAUSI_unifiedv3_BSVp1-7_withbad3D_TBDsplit_20240406.csv")

# Replace missing <ulcer_length>, <ulcer_width>, <ulcer_depth> with 0
df.loc[df['ulcer_length'].isna(), 'ulcer_length'] = 0
df.loc[df['ulcer_width'].isna(), 'ulcer_width'] = 0
df.loc[df['ulcer_depth'].isna(), 'ulcer_depth'] = 0

# 292-022 post_debridement depth -> 0.2 cm
df.loc[df['subject_number'] == '292-022', 'ulcer_depth'] = 0.2

# Exclude subjects 202-034, 203-009 from dataset
df = df[~df['subject_number'].isin(['202-034', '203-009'])].reset_index(drop = True)

df.to_csv("/home/efs/ziping/workspaces/dfu/clf_algo_release_202404/src/data/WAUSI_unifiedv3_BSVp1-7_withbad3D_TBDsplit_20240406_modified.csv", index = False)