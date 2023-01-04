#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@License: (C) Copyright 2013-2022.
@Desc:

"""

import numpy as np
import pandas as pd
import cv2
import shapely.geometry as shp
from scipy.interpolate import splprep, splev


def crt_frac_map_updated_cell_type_frac(adata, target_ct='AT2_like'):
    """
    create a np array using coordinates from adata.obsm['spatial'] with fraction 
    values returned by cell2location 
    And scale the fraction into [0, 255]
    """
    coords = adata.obsm['spatial'] # np.ndarray
    x_coord = coords[:,0] # width
    y_coord = coords[:,1] # hight
    
    def cvt_range(old_arr, new_min=0, new_max=255):
        old_max, old_min = old_arr.max(), old_arr.min()
        old_rng = old_max - old_min
        new_rng = new_max - new_min
        return (((old_arr - old_min) * new_rng) / old_rng) + new_min
    
    im_map = np.zeros((max(y_coord)+1,max(x_coord)+1)) # 255: white
    ct_update = adata.obsm['cell_type_update']
    im_map[y_coord, x_coord] = cvt_range(
        ct_update.apply(lambda x: x[target_ct] if x['ct_update'] == target_ct else 0, axis=1).values
    ) 
    
    return im_map.astype(np.uint8)


def find_boundary(im_map_arr, 
                 min_thre_binary=None, 
                 max_thre_binary=None, 
                 gaussian_sigma=0, 
                 canny_thre1=200, 
                 canny_thre2=255):
    """
    find all contuors using open cv
    All contours are returned by sorting their areas, descreasing. 
    
    return
    -----
    contuors areas and contuors objects 
    
    """
    
    a = im_map_arr.max()
    if min_thre_binary is None:
        min_thre_binary = a/2
    if max_thre_binary is None:
        max_thre_binary = a
    
    _, thresh = cv2.threshold(im_map_arr, 
                              min_thre_binary, 
                              max_thre_binary, 
                              cv2.THRESH_BINARY)#cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    blurred = cv2.GaussianBlur(thresh, (3, 3), gaussian_sigma)
    tight_edged = cv2.Canny(blurred, canny_thre1,canny_thre2)
    
    # find the contours in the dilated image
    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # apply the dilation operation to the edged image
    dilate = cv2.dilate(tight_edged, kernel, iterations=2) # 
    # find the contours in the dilated image
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        raise ValueError('No boundary found')
    
    r_areas = [cv2.contourArea(c) for c in contours]
    sorted_area = np.sort(r_areas)[::-1]
    sorted_area_ind = np.argsort(r_areas)[::-1]
    
    sorted_contour = []
    for i in sorted_area_ind:
        sorted_contour.append(contours[i])
    
    return sorted_area, sorted_contour


def smooth_boundary(boundary, factor=0.01,intp=False, n_interp=200):
    """
    smooth all the boundarys
    
    parameters
    -----
    boudarys: list
        list of countours returned by cv2.findContours
    
    factor: float
        smoothing factor, the bigger, the smoother
    
    """
    smoothened=[]
    for c in boundary:
        epsilon = factor*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        # smoothened.append(approx)
        
        if intp:
            # https://agniva.me/scipy/2016/10/25/contour-smoothing.html
            x,y = approx.T
            # Convert from numpy arrays to normal arrays
            x = x.tolist()[0]
            y = y.tolist()[0]
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
            tck, u = splprep([x,y], u=None, s=1, per=1)
            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
            u_new = np.linspace(u.min(), u.max(), n_interp)
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
            x_new, y_new = splev(u_new, tck, der=0)
            # Convert it back to numpy format for opencv to be able to display it
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
            smoothened.append(np.asarray(res_array, dtype=np.int32))
        else:
            smoothened.append(approx) 
        
    return smoothened


def find_distance_boundary(boundary, distance=5):
    """
    find the ploy of boudary by given mboundary and distance
    
    parameters
    -----
    boundary: np.ndarray
        an element of contuors list returned by cv2.findContours,
        the element is 3d np array.
    distance: float
    
    return shapely.geometry ploygons
    """
    # o_c = f_contour[0].squeeze() #[:,0]:x, [:,1]:y
    o_c = boundary.squeeze() #[:,0]:x, [:,1]:y
    # Create a Polygon from the nx2 array 
    oc_poly = shp.Polygon(o_c)
    
    # Create offset airfoils, both inward and outward
    poffpoly = oc_poly.buffer(distance, single_sided=True)  # Outward offset
    noffpoly = oc_poly.buffer(-distance, single_sided=True)  # Inward offset 1000 um
    
    return oc_poly, poffpoly, noffpoly
