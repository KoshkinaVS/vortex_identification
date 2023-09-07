
# from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA
# from scipy import signal

import time

from tqdm import tqdm

import datetime
from datetime import timedelta

# from scipy.ndimage import filters
from tqdm import tqdm

# from load_data import *

path_dir = '/storage/NAADSERVER/NAAD/HiRes/PressureLevels/'

g = 9.80665


# нормируем критерий к [0,1]
def crit_log(Crit):
    Crit[Crit <= 0] = None
    log = np.log10(Crit)
    log_2 = (log - np.nanmin(log))/(np.nanmax(log) - np.nanmin(log))
    return log_2

# стандартизуем критерий к [-1,1]
def crit_stand(Crit):
    log_2 = (Crit - np.nanmean(Crit))/(np.std(Crit))
    return log_2

# считаем детерминант для инварианта R
def my_det(grad_tensor):
    a, b, c = grad_tensor[0,0], grad_tensor[0,1], grad_tensor[0,2]
    d, e, f = grad_tensor[1,0], grad_tensor[1,1], grad_tensor[1,2]
    g, h, i = grad_tensor[2,0], grad_tensor[2,1], grad_tensor[2,2]
    
    det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h
    return det

# считаем завихренность, omega.shape = (time,level,lat,lon,3)
def compute_omega(ds):    
    vx = np.gradient(ds.ve, dist_m, axis=3)
    uy = np.gradient(ds.ue, dist_m, axis=2)
    
    omega_z = vx - uy

    wy = np.gradient(ds.w, dist_m, axis=2)
    vz = np.gradient(ds.ve, 1., axis=1)/dz
    
    uz = np.gradient(ds.ue, 1., axis=1)/dz
    wx = np.gradient(ds.w, dist_m, axis=3)
    
    omega_x = wy - vz
    omega_y = uz - wx
    
    omega = np.array([omega_x, omega_y, omega_z])
    omega = omega.transpose(1, 2, 3, 4, 0)
    
    return omega

# считаем тензор градиента скорости, grad_tensor.shape = (3,3,time,level,lat,lon)
def compute_grad_tensor(u, v, w, dist_m, dz):
    true_ux, true_uy, true_uz = np.gradient(u, dist_m, dist_m, 1., axis=[3,2,1])
    true_vx, true_vy, true_vz = np.gradient(v, dist_m, dist_m, 1., axis=[3,2,1])
    true_wx, true_wy, true_wz = np.gradient(w, dist_m, dist_m, 1., axis=[3,2,1])

    true_uz = true_uz/dz
    true_vz = true_vz/dz
    true_wz = true_wz/dz
    
    # собрать в тензор
    grad_tensor = np.array([
                            [true_ux, true_uy, true_uz], 
                            [true_vx, true_vy, true_vz], 
                            [true_wx, true_wy, true_wz],
                       ])
    
    return grad_tensor

# считаем 2d тензор градиента скорости, grad_tensor.shape = (2,2,...,lat,lon)
def compute_grad_tensor_2d(u, v, dist_m):
    true_ux, true_uy = np.gradient(u, dist_m, dist_m, 1., axis=[-1,-2])
    true_vx, true_vy = np.gradient(v, dist_m, dist_m, 1., axis=[-1,-2])
    
    # собрать в тензор
    grad_tensor_2d = np.array([
                            [true_ux, true_uy, ], 
                            [true_vx, true_vy, ], 
                       ])
    
    return grad_tensor_2d

# считаем тензор скоростей деформации S и тензор завихренности A
def compute_S_A(grad_tensor):
    s12 = 0.5*(grad_tensor[0,1] + grad_tensor[1,0])
    s13 = 0.5*(grad_tensor[0,2] + grad_tensor[2,0])
    s23 = 0.5*(grad_tensor[1,2] + grad_tensor[2,1])
    
    S = np.array([
                [grad_tensor[0,0], s12, s13], 
                [s12, grad_tensor[1,1], s23], 
                [s13, s23, grad_tensor[2,2]],
             ])
    
    a12 = 0.5*(grad_tensor[0,1] - grad_tensor[1,0])
    a13 = 0.5*(grad_tensor[0,2] - grad_tensor[2,0])
    a23 = 0.5*(grad_tensor[1,2] - grad_tensor[2,1])
    
    diag_0 = np.zeros(shape=(grad_tensor.shape[2], grad_tensor.shape[3], grad_tensor.shape[4], grad_tensor.shape[5]))

    A = np.array([
                [diag_0, a12, a13], 
                [-a12, diag_0, a23], 
                [-a13, -a23, diag_0],
             ])
    
    return S, A

# считаем Q-критерий
def compute_Q(S, A, normalize=True):

    norm_S = LA.norm(S, axis = (0,1))
    norm_A = LA.norm(A, axis = (0,1))

    Q = 0.5*(norm_A*norm_A - norm_S*norm_S)
    if normalize:
        Q = crit_log(Q)
        print('Q computed')
    
    return Q

# считаем delta-критерий
def compute_delta(grad_tensor, S, A, normalize=True):

    R = my_det(grad_tensor)
    Q = compute_Q(S, A, normalize=False)

    delta = (Q/3)**3 + (0.5*R)**2
    if normalize:
        delta = crit_log(delta)
        
    print('delta computed')
    return delta

# считаем lambda2-критерий
def compute_lambda2(S, A, normalize=True):
    S = S.transpose(2, 3, 4, 5, 0, 1)
    A = A.transpose(2, 3, 4, 5, 0, 1)
    S = np.matmul(S, S)
    A = np.matmul(A, A)
    SA = S + A

    lambda2 = np.zeros(shape=(SA.shape[0], SA.shape[1], SA.shape[2], SA.shape[3]))

    # TODO: for only 1 level
    for f in tqdm(range(SA.shape[0])):
        for i in range(SA.shape[1]):
            for j in range(SA.shape[2]):
                for k in range(SA.shape[3]):
                    if np.isnan(SA[f,i,j,k,1,1]) == True:
                        lambda2[f,i,j,k] = np.nan
                    else:
                        lambd = LA.eigvalsh(SA[f,i,j,k])
                        lambda2[f,i,j,k] = lambd[1]       

    lambda2 = -lambda2
    if normalize:
        lambda2 = crit_log(lambda2)
    return lambda2

# считаем мнимую часть СЗ градиента скорости lambda_ci и действительнозначный СВ - отвечает за ось вращения
def compute_swirling_strength(grad_tensor):
    
    grad_tensor = grad_tensor.transpose(2, 3, 4, 5, 0, 1)
    
    start = time.time() ## точка отсчета времени

    eigenvalues = np.zeros(shape=(grad_tensor.shape[0], grad_tensor.shape[1], grad_tensor.shape[2], grad_tensor.shape[3], 3), dtype = 'complex_')
    sw_vec_reoredered = np.zeros(shape=(grad_tensor.shape[0], grad_tensor.shape[1], grad_tensor.shape[2], grad_tensor.shape[3], 3))
    
    
    for f in tqdm(range(grad_tensor.shape[0])):
        for i in range(grad_tensor.shape[1]):
            for j in range(grad_tensor.shape[2]):
                for k in range(grad_tensor.shape[3]):
                    if np.isnan(grad_tensor[f,i,j,k]).any() == True:
                        eigenvalues[f,i,j,k,:] = np.nan
                        sw_vec_reoredered[f,i,j,k] = np.nan
                        
                    else:
                        lambd, eigv = LA.eig(grad_tensor[f,i,j,k])
                        eigenvalues[f,i,j,k] = np.array(lambd, dtype = "complex_")
                
                        order = np.argsort(np.imag(eigenvalues[f,i,j,k]))
                        sw_vec_reoredered[f,i,j,k] = np.array(eigv)[:,order][:,1]
                        
                        
    end = time.time() - start ## собственно время работы программы
    print(f'swirling_strength computed ({end/60} min)') ## вывод времени
    
    only_im = eigenvalues - np.real(eigenvalues) ## выделение мнимой части СЗ
    sw_str = np.abs(np.imag(only_im)[:,:,:,:,1])
    sw_str[sw_str <= 0.] = np.nan ## критерий > 0
    
    return sw_str, sw_vec_reoredered


# считаем Rortex-критерий, Rortex > 0 - циклон, < 0 - АЦ
def compute_rortex(sw_str, sw_vec, omega):
    scalar = np.einsum('ijlmk,ijlmk->ijlm', omega, sw_vec)
#     sign_u_r = np.where(np.repeat(scalar[:,:,:,:,np.newaxis], 3, axis=4) > 0, sw_vec, -sw_vec)
    scalar_sq = scalar*scalar
#     scalar_sq[scalar_sq < 1e-16] = None
    im_omega_part = 4*sw_str*sw_str/(scalar*scalar)
#     sign_u_r_simple = np.where(scalar[:,:,:,:] > 0, 1, -1)
    
    R = (1-np.sqrt(1-im_omega_part))*scalar
    return R