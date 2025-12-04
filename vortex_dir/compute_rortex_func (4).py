import pandas as pd 
import numpy as np
import math
import xarray as xr
from numpy import linalg as LA

import time

from tqdm import tqdm

import datetime
from datetime import timedelta

from tqdm import tqdm

from geopy.distance import geodesic

def calculate_distances(latitudes, longitudes):
    """
    Вычисляет расстояния между точками на широтно-долготной сетке.
    Возвращает расстояния в метрах по широте и долготе.
    """
    # Для долготы: расстояние между точками на экваторе (зависит от широты)
    lon_dist = geodesic((latitudes[0], longitudes[0]), (latitudes[0], longitudes[1])).meters
    # Для широты: расстояние между точками постоянно
    lat_dist = geodesic((latitudes[0], longitudes[0]), (latitudes[1], longitudes[0])).meters
    return lat_dist, lon_dist


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
def compute_omega(u, v, w, dist_m, dz):    
    vx = np.gradient(v, dist_m, axis=3)
    uy = np.gradient(u, dist_m, axis=2)
    
    omega_z = vx - uy

    wy = np.gradient(w, dist_m, axis=2)
    vz = np.gradient(v, 1., axis=1)/dz
    
    uz = np.gradient(u, 1., axis=1)/dz
    wx = np.gradient(w, dist_m, axis=3)
    
    omega_x = wy - vz
    omega_y = uz - wx
    
    omega = np.array([omega_x, omega_y, omega_z])
    omega = omega.transpose(1, 2, 3, 4, 0)
    
    return omega

# считаем 2d завихренность (z-компонента), omega.shape = (time,level,lat,lon,1)
def compute_omega_2d(u, v, dist_m):    
    vx = np.gradient(v, dist_m, axis=-1)
    uy = np.gradient(u, dist_m, axis=-2)
    
    omega_z = vx - uy
    
    return omega_z

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
    true_ux, true_uy = np.gradient(u, dist_m, dist_m, axis=[-1,-2])
    true_vx, true_vy = np.gradient(v, dist_m, dist_m, axis=[-1,-2])
    
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

# считаем тензор скоростей деформации S и тензор завихренности A
def compute_S_A_2d(grad_tensor):
    s12 = 0.5*(grad_tensor[0,1] + grad_tensor[1,0])
    
    S = np.array([
                [grad_tensor[0,0], s12], 
                [s12, grad_tensor[1,1]], 
             ])
    
    a12 = 0.5*(grad_tensor[0,1] - grad_tensor[1,0])
    
    diag_0 = np.zeros(shape=(grad_tensor.shape[2], grad_tensor.shape[3], grad_tensor.shape[4], grad_tensor.shape[5]))
#     diag_0 = np.zeros(shape=(grad_tensor.shape[2], grad_tensor.shape[3], grad_tensor.shape[4]))
    

    A = np.array([
                [diag_0, a12], 
                [-a12, diag_0], 
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
def compute_delta(grad_tensor, S, A, normalize=True, case='3d'):

    if case == '2d':
        R = grad_tensor[0,0]*grad_tensor[1,1] - grad_tensor[1,0]*grad_tensor[0,1]
    else:
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
    print(f'3d swirling_strength computed ({end/60} min)') ## вывод времени
    
    only_im = eigenvalues - np.real(eigenvalues) ## выделение мнимой части СЗ
    sw_str = np.abs(np.imag(only_im)[:,:,:,:,1])
    sw_str[sw_str <= 0.] = np.nan ## критерий > 0
    
    return sw_str, sw_vec_reoredered

def compute_swirling_strength(grad_tensor):
    """
    Векторизованное вычисление swirling strength (λ_ci) и собственных векторов
    с обработкой NaN и проверкой входных данных.
    """
    # Перенос осей тензора в конец (time, level, lat, lon, 3, 3)
    grad_tensor = grad_tensor.transpose(2, 3, 4, 5, 0, 1)
    
    # Проверка размерностей
    if grad_tensor.size == 0:
        empty_shape = grad_tensor.shape[:-2]
        return np.full(empty_shape, np.nan), np.full(empty_shape + (3,), np.nan)
    
    # Развертывание массива для векторизованных операций
    flat_grad = grad_tensor.reshape(-1, 3, 3)
    n_points = flat_grad.shape[0]
    
    # Маска для валидных точек (без NaN)
    valid_mask = ~np.isnan(flat_grad).any(axis=(1, 2))
    
    # Выделение памяти для результатов
    eigenvalues = np.full((n_points, 3), np.nan + 1j*np.nan, dtype=np.complex128)
    eigenvectors = np.full((n_points, 3, 3), np.nan, dtype=np.complex128)
    
    # Вычисление только для валидных точек
    if np.any(valid_mask):
        eigvals, eigvecs = np.linalg.eig(flat_grad[valid_mask])
        eigenvalues[valid_mask] = eigvals
        eigenvectors[valid_mask] = np.moveaxis(eigvecs, -1, 1)  # Изменение порядка осей
    
    # Восстановление исходной формы
    original_shape = grad_tensor.shape[:-2]
    eigenvalues = eigenvalues.reshape(original_shape + (3,))
    eigenvectors = eigenvectors.reshape(original_shape + (3, 3))
    
    # Сортировка по мнимой части
    im_eig = np.imag(eigenvalues)
    sort_idx = np.argsort(im_eig, axis=-1)
    
    # Выбор нужного собственного вектора
    sw_vec = np.take_along_axis(
        eigenvectors, 
        sort_idx[..., np.newaxis, :], 
        axis=-1
    )[..., 1]
    
    # Вычисление swirling strength
    sw_str = np.abs(im_eig[..., 1])
    sw_str[sw_str <= 0] = np.nan
    
    return sw_str, sw_vec

# считаем мнимую часть СЗ градиента скорости lambda_ci для 2d 
def compute_swirling_strength_2d(grad_tensor):
    
    grad_tensor = grad_tensor.transpose(2, 3, 4, 5, 0, 1)
    
    start = time.time() ## точка отсчета времени

    eigenvalues = np.zeros(shape=(grad_tensor.shape[0], grad_tensor.shape[1], grad_tensor.shape[2], grad_tensor.shape[3], 2), dtype = 'complex_')
    
    
    for f in tqdm(range(grad_tensor.shape[0])):
        for i in tqdm(range(grad_tensor.shape[1])):
            for j in range(grad_tensor.shape[2]):
                for k in range(grad_tensor.shape[3]):
                    if np.isnan(grad_tensor[f,i,j,k]).any() == True:
                        eigenvalues[f,i,j,k,:] = np.nan                        
                    else:
                        lambd, eigv = LA.eig(grad_tensor[f,i,j,k])
                        eigenvalues[f,i,j,k] = np.array(lambd, dtype = "complex_")
                  
    end = time.time() - start ## собственно время работы программы
    # print(f'2d swirling_strength computed ({end/60} min)') ## вывод времени 
    
    only_im = eigenvalues - np.real(eigenvalues) ## выделение мнимой части СЗ
    sw_str = np.abs(np.imag(only_im)[:,:,:,:,1])
    sw_str[sw_str <= 0.] = np.nan ## критерий > 0
    
    return sw_str


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

def compute_rortex(sw_str, omega, sw_vec):
    scalar = np.einsum('...i,...i->...', omega, sw_vec)

    im_omega_part = 4 * sw_str**2 / (scalar**2 + 1e-16)  # добавлен малый делитель для стабильности
    
    # Вычисление Rortex
    R = (1 - np.sqrt(1 - im_omega_part)) * scalar
    return R

# считаем 2d Rortex-критерий, Rortex > 0 - циклон, < 0 - АЦ
def compute_rortex_2d(sw_str_2d, omega_2d):
    R_2d = (1-np.sqrt(1-4*sw_str_2d*sw_str_2d/(omega_2d*omega_2d)))*omega_2d
    return R_2d


def get_R2D_ds(u, v, dist_m):
    ################ 2d расчет тензора градиента скорости и завихренности ###################
    grad_tensor_2d = compute_grad_tensor_2d(u, v, dist_m)
    omega_2d = compute_omega_2d(u, v, dist_m)

    ################ 2d расчет swirling_strength и rortex ###################
    sw_str_2d = compute_swirling_strength_2d(grad_tensor_2d)
    R_2d = compute_rortex_2d(sw_str_2d, omega_2d)
    return R_2d

def get_R2D_ds_from_grads(du_dx, du_dy, dv_dx, dv_dy):  # Только нужные градиенты
    omega_2d = dv_dx - du_dy  # Прямое вычисление без лишних операций
    
    # Оптимизированное создание тензора
    grad_tensor_2d = np.stack([
        np.stack([du_dx, du_dy], axis=-1),
        np.stack([dv_dx, dv_dy], axis=-1)
    ], axis=-2)
    
    # Упрощенное вычисление swirling strength для 2D
    trace = du_dx + dv_dy
    det = du_dx*dv_dy - du_dy*dv_dx
    discriminant = trace**2 - 4*det
    sw_str_2d = np.sqrt(np.maximum(-discriminant, 0))/2
    sw_str_2d = np.where(sw_str_2d <= 0, np.nan, sw_str_2d)
    
    # Упрощенный расчет Rortex
    with np.errstate(divide='ignore', invalid='ignore'):
        term = 4 * sw_str_2d**2 / (omega_2d**2 + 1e-16)
        R_2d = (1 - np.sqrt(np.maximum(1 - term, 0))) * omega_2d
        R_2d = np.where(np.isclose(omega_2d, 0), np.nan, R_2d)  # Обработка деления на 0
    
    return R_2d


# считаем завихренность, omega.shape = (time,level,lat,lon,3)
def compute_omega_from_grads(du_dx, du_dy, du_dz, 
                dv_dx, dv_dy, dv_dz,
                dw_dx, dw_dy, dw_dz):
    
    omega_z = dv_dx - du_dy
    
    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    
    omega = np.array([omega_x, omega_y, omega_z])
    omega = omega.transpose(1, 2, 3, 4, 0)
    
    return omega

def get_R3D_ds_from_grads(du_dx, du_dy, du_dz,
                   dv_dx, dv_dy, dv_dz,
                   dw_dx, dw_dy, dw_dz):

    # # считаем завихренность, omega.shape = (time,level,lat,lon,3)
    # def compute_omega(du_dx, du_dy, du_dz, 
    #                 dv_dx, dv_dy, dv_dz,
    #                 dw_dx, dw_dy, dw_dz):
        
    #     omega_z = dv_dx - du_dy
        
    #     omega_x = dw_dy - dv_dz
    #     omega_y = du_dz - dw_dx
        
    #     omega = np.array([omega_x, omega_y, omega_z])
    #     omega = omega.transpose(1, 2, 3, 4, 0)
        
    #     return omega


    omega = compute_omega_from_grads(du_dx, du_dy, du_dz, 
                    dv_dx, dv_dy, dv_dz,
                    dw_dx, dw_dy, dw_dz)

    grad_tensor = np.array([[du_dx, du_dy, du_dz],
                        [dv_dx, dv_dy, dv_dz],
                        [dw_dx, dw_dy, dw_dz]])
    
    ################ 2d расчет swirling_strength и rortex ###################
    sw_str, sw_vec = compute_swirling_strength(grad_tensor)
    R = compute_rortex(sw_str, sw_vec, omega)
    return R
