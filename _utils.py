#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@License: (C) Copyright 2013-2022.
@File: _utils.py
@Desc:

"""

import numpy as np
import pandas as pd
import scanpy as sc

from matplotlib import cm, colors


def get_adata(adata_path, log=False):
    """
    adata_path: str
        adata contains with results from cell2location
    """
    adata = sc.read_h5ad(adata_path)
    if log:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata


def update_celltype(adata, ct_list, min_frac_ct=1.5,update_cell2loc=False):
    """
    update cell type by cell2location according to ratio between first and second cell type fraction
    """
    df_ct_frac = adata.obs[ct_list].copy()
    df_ct_frac = df_ct_frac.div(df_ct_frac.sum(axis=1), axis=0)
    ct_frac_arr = df_ct_frac.values
    ind_sec_frt = ct_frac_arr.argsort()[:,-2:]
    df_ct_frac['sec_max_ind'] = ind_sec_frt[:,0]
    df_ct_frac['top_max_ind'] = ind_sec_frt[:,1]
    df_ct_frac['ratio']=df_ct_frac.apply(lambda x: x[ct_list[int(x['top_max_ind'])]]/x[ct_list[int(x['sec_max_ind'])]], axis=1)
    # adata.obs[f'{target_ct}_binary'] = df_ct_frac.apply(lambda x: 'Yes' if x['top_max_ind'] == ct_list.index(target_ct) and x['ratio']>min_frac_ct else 'No', axis=1)
    df_ct_frac['ct_update'] = df_ct_frac.apply(lambda x: ct_list[int(x['top_max_ind'])] if x['ratio'] > min_frac_ct else 'NOT_SPEC', axis=1)
    if update_cell2loc:
        adata.obs['cell2location_max'] = df_ct_frac.apply(lambda x: ct_list[int(x['top_max_ind'])], axis=1)
    
    adata.obsm['cell_type_update'] = df_ct_frac

def set_cell2loc_palette(ct_list):
    vega_10 = list(map(colors.to_hex, cm.tab10.colors))
    vega_10_scanpy = vega_10.copy()
    vega_10_scanpy[2] = '#279e68'  # green
    vega_10_scanpy[4] = '#aa40fc'  # purple
    vega_10_scanpy[8] = '#b5bd61'  # kakhi

    vega_20 = list(map(colors.to_hex, cm.tab20.colors))
    # reorderd, some removed, some added
    vega_20_scanpy = [
        # dark without grey:
        *vega_20[0:14:2],
        *vega_20[16::2],
        # light without grey:
        *vega_20[1:15:2],
        *vega_20[17::2],
        # manual additions:
        '#ad494a',
        '#8c6d31'
    ]
    vega_20_scanpy[2] = vega_10_scanpy[2]
    vega_20_scanpy[4] = vega_10_scanpy[4]
    vega_20_scanpy[7] = vega_10_scanpy[8]  # kakhi shifted by missing grey
    # TODO: also replace pale colors if necessary
    default_20 = vega_20_scanpy

    len_ct = len(ct_list)
    return dict(zip(ct_list, default_20[:len_ct]))

