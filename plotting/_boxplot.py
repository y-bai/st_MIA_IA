"""
@Author: Yong Bai, yong.bai@hotmail.com
@License: (C) Copyright 2013-2022.
@Desc:

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator


def _boxplot(data, x, y, ax, comp_pairs,
             orient='v', orders=None,fontsize=8,
             showfliers=False, linewidth=0.5, width=0.8, gene=False,
             bg_f = ['#b9f2f0', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff', '#debb9b', '#fab0e4', '#fffea3'],
            ln_f = ['#00d7ff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#ffc400']):
    
    if orders is None:
        orders = data.x.unique()
    
    # from matplotlib import cm, colors
    # x = list(map(colors.to_hex, sns.color_palette('pastel')))
    # print(x)
    # x = list(map(colors.to_hex, sns.color_palette('bright')))
    # print(x)
    
    # light color from 'pastel'
    # bg_f = ['#b9f2f0', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff', '#debb9b', '#fab0e4', '#fffea3']
    # bright color from 'bright'
    # ln_f = ['#00d7ff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#ffc400']
        
    palette = dict(zip(orders, bg_f))
        
    plot_param={
        'data':data,
        'x':x,
        'y':y,
        'order':orders,
        'orient':orient
    }
    
    axa = None
    annot = None
    if not gene:
        axa = sns.boxplot(**plot_param,
                          ax=ax,palette=palette, 
                          showfliers=showfliers, 
                          linewidth=linewidth,
                          width=width)
    
        for i,artist in enumerate(ax.patches): #In matplotlib 3.5 the boxes are stored in ax.patches instead of ax.artists.
            # Set the linecolor on the artist to the facecolor, and set the facecolor to None
            # col = artist.get_facecolor()
            # artist.set_edgecolor(col)
            # artist.set_facecolor('None')
            artist.set_edgecolor(ln_f[i])
    
            # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
            # Loop over them here, and use the same colour as above
            for j in range(i*5,i*5+5):
                line = axa.lines[j]
                # line.set_color(col)
                # line.set_mfc(col)
                # line.set_mec(col)

                line.set_color(ln_f[i])
                line.set_mfc(ln_f[i])
                line.set_mec(ln_f[i])
                
        annot = Annotator(axa, pairs=comp_pairs, **plot_param)
    else:
        axa = sns.violinplot(**plot_param,
                          ax=ax,palette=palette, 
                          linewidth=linewidth,
                          width=width, inner=None,cut=0)
        for i,artist in enumerate(ax.collections): 
            artist.set_edgecolor(ln_f[i])
        annot = Annotator(axa, pairs=comp_pairs, **plot_param, plot="violinplot")
        
    # https://github.com/trevismd/statannotations/blob/master/usage/example.ipynb
    annot.configure(test='Mann-Whitney', comparisons_correction="BH", text_format='star', 
                    line_width=linewidth,fontsize=fontsize,loc='inside',line_height=0, text_offset=0.5)
    annot.apply_and_annotate()
    
    return axa