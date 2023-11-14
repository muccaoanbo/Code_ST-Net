# 计算ROI区域内的metrics
import argparse
import datetime
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model.network_model_ST import CNN_ST
from utils.evaluate import concordance_correlation_coefficient as ccc
from utils.evaluate import KL
from utils.dataset_ST import patients_dataset, Patient

import toml
from easydict import EasyDict
import nibabel as nib
import csv
import pandas as pd
import scipy.io as scio
from scipy import ndimage
import scipy.stats
import skimage
from skimage import metrics
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import matplotlib.pyplot as plt

def roi_metrics(name):
    res_x = []
    res_y = []

    k_trans_x = []
    k_trans_y = []

    img_name = name + '_Tmax.nii.gz'  # 此处的name应该是传入的参数为ID
    # img是真实的ktrans参数图，是三维的，且未做平滑处理的
    img = nib.load(os.path.join('/data0/BME_caoanbo/project_coding/K-trans_STNet/data/raw/params_High', name, img_name))
    # test_img_name是模型预测得到的值
    test_img_name = 'Fold1_' + name + '_test_Tmax.nii.gz'
    test_img = nib.load(os.path.join('/data0/BME_caoanbo/project_coding/K-trans_STNet_version1/ST/exp_3_physic_v6/0830_ST_OP_NM_CAhigh_1_2_T8V2_MAE_lr0.0001_batch512_InRawHighDCE_GtRaw_KtransTh_Adam_10subjects_5folds_T10norm_Aug/test',
                                     test_img_name)).get_fdata()  # 模型预测得到的值
    GT_img = img.get_fdata()  # 金标准的值



    # 预测的cbv的值如果小于0，将其置为0
    test_img[test_img < 0] = 0
    # PredictionImg = nib.Nifti1Image(test_img, img.affine, img.header)  # 存储成3d.nii.gz数据时对应的格式
    # 读入brain mask
    # BrainMask = nib.load(os.path.join('/data0/BME_caoanbo/project_coding/K-trans_STNet/data/raw/BrainMask',
    #                                   name + '.nii.gz')).get_fdata()
    # penumbra区域的mask，计算ROI内的metrics
    BrainMask = nib.load(os.path.join('/data0/BME_caoanbo/project_coding/K-trans_STNet/data/raw/vpenumbra6',
                                      name + '_vpenumbra6.nii.gz')).get_fdata()
    test_WB = test_img * BrainMask  # 乘以mask，只考虑脑组织区域的
    GT_WB = GT_img * BrainMask

    #### 先不做滤波处理
    # kernel_size = (3,3,3) # 平滑处理
    # test_WB_smoothed = ndimage.median_filter(test_WB, size = kernel_size) # 滤波
    # GT_WB_smoothed = ndimage.median_filter(GT_WB, size = kernel_size)
    # GT_WB_smoothed[GT_WB_smoothed > 0.05] = 0

    test_WB_smoothed = test_WB
    GT_WB_smoothed = GT_WB

    GT_WB_max = np.max(GT_WB_smoothed[BrainMask == 1])
    test_WB_max = np.max(test_WB_smoothed[BrainMask == 1])

    GT_WB_smoothed = GT_WB_smoothed  # / GT_WB_max
    test_WB_smoothed = test_WB_smoothed  # / test_WB_max

    th = 0
    # INCRESE TH TO 0.2 AND DO COMPARISON (for opening area)
    GT_WB_smoothed[GT_WB_smoothed < th] = 0  # cbv的值不可能小于0
    # GT_WB_smoothed[GT_WB_smoothed > 100] = 0
    test_WB_smoothed[test_WB_smoothed < th] = 0
    # test_WB_smoothed[test_WB_smoothed > 100] = 0

    GT_WB_smoothed_array = GT_WB_smoothed[BrainMask == 1]  # 只考虑brain mask区域，这时变成了一维数组
    test_WB_smoothed_array = test_WB_smoothed[BrainMask == 1]

    # 计算评估参数
    Prediction_vs_GT_SCC, p_value = scipy.stats.spearmanr(test_WB_smoothed_array, GT_WB_smoothed_array,
                                                          nan_policy='omit')
    Prediction_vs_GT_PCC, p_value = np.corrcoef(GT_WB_smoothed_array, test_WB_smoothed_array)[1]
    Prediction_vs_GT_ccc = ccc(GT_WB_smoothed_array, test_WB_smoothed_array)
    Prediction_vs_GT_nrmse = nrmse(GT_WB_smoothed_array, test_WB_smoothed_array, normalization='euclidean')
    Prediction_vs_GT_KL = KL(test_WB_smoothed_array, GT_WB_smoothed_array, BrainMask, 128)  # KL散度

    Prediction_vs_GT_PSNR_sum = 0
    Prediction_vs_GT_SSIM_sum = 0
    slice_count = 0
    for i in range(test_img.shape[2]):  # slice维度上
        GT_WB_smoothed_slice = GT_WB_smoothed[:, :, i]
        test_WB_smoothed_slice = test_WB_smoothed[:, :, i]
        BrainMask_slice = BrainMask[:, :, i]

        # 先对切片中的信号强度值做归一化，因为psnr和ssim计算的时候对值的范围归一化到了-1到1
        # 为防止分母为0，需要平滑处理
        # print(GT_WB_smoothed_slice.size)
        GT_WB_smoothed_slice = (GT_WB_smoothed_slice - np.min(GT_WB_smoothed_slice)) / (
                    (np.max(GT_WB_smoothed_slice) - np.min(GT_WB_smoothed_slice)) + 1e-6)
        test_WB_smoothed_slice = (test_WB_smoothed_slice - np.min(test_WB_smoothed_slice)) / (
                    (np.max(test_WB_smoothed_slice) - np.min(test_WB_smoothed_slice)) + 1e-6)

        # 原始的mask中在某些slice上没有对应的脑组织，不存在脑组织区域，如果体素点小于SSIM计算时的窗宽，则不考虑
        if len(GT_WB_smoothed_slice[BrainMask_slice == 1]) <= 7 or len(
                test_WB_smoothed_slice[BrainMask_slice == 1]) <= 7:
            continue
        slice_psnr = PSNR(GT_WB_smoothed_slice[BrainMask_slice == 1], test_WB_smoothed_slice[BrainMask_slice == 1])
        slice_ssim = SSIM(GT_WB_smoothed_slice[BrainMask_slice == 1], test_WB_smoothed_slice[BrainMask_slice == 1])

        Prediction_vs_GT_PSNR_sum += slice_psnr
        Prediction_vs_GT_SSIM_sum += slice_ssim
        slice_count += 1  # 记录多少个slice参与了psnr和ssim的计算，因为有些slice上没有脑组织，不用考虑进来
    Prediction_vs_GT_PSNR = Prediction_vs_GT_PSNR_sum / slice_count  # 平均值
    Prediction_vs_GT_SSIM = Prediction_vs_GT_SSIM_sum / slice_count

    # Prediction_vs_GT_SSIM = SSIM(GT_WB_smoothed_array, test_WB_smoothed_array)
    # Prediction_vs_GT_SSIM = 0
    # for i in range(18):
    # 	t = test_WB_smoothed[i]
    # 	g = GT_WB_smoothed[i]
    # 	b = BrainMask[i]
    # 	Prediction_vs_GT_SSIM = SSIM(t[b==1], g[b==1])#, datarange=test_WB_smoothed_array.max()-test_WB_smoothed_array.min())
    # 	print(Prediction_vs_GT_SSIM)

    loss_dict = {'SCC': Prediction_vs_GT_SCC, 'PCC': Prediction_vs_GT_PCC, 'CCC': Prediction_vs_GT_ccc,
                 'NRMSE': Prediction_vs_GT_nrmse, 'KL': Prediction_vs_GT_KL, 'PSNR': Prediction_vs_GT_PSNR,
                 'SSIM': Prediction_vs_GT_SSIM}
    loss_info = '{}, SCC:{}, PCC:{}, CCC:{}, NRMSE:{}, KL:{}, PSNR:{}, SSIM:{}'.format(
        name, Prediction_vs_GT_SCC, Prediction_vs_GT_PCC, Prediction_vs_GT_ccc, Prediction_vs_GT_nrmse,
        Prediction_vs_GT_KL, Prediction_vs_GT_PSNR, Prediction_vs_GT_SSIM
    )

    return loss_info, loss_dict


loss_all = {}
roi_test_path = '/data0/BME_caoanbo/project_coding/K-trans_STNet/data/raw/openingMask_High_test'
for roi_sub in os.listdir(roi_test_path):
    roi_sub_id = roi_sub[0:15]
    pat_result, loss_dict = roi_metrics(roi_sub_id)
    loss_all[roi_sub_id] = loss_dict

print('------------Test Result------------')

# 存储测试时计算的评估指标的值
df = pd.DataFrame(loss_all).T
loss_all['Avg'] = {}
for k, v in df.items():
    avg = v.mean()
    loss_all['Avg'][k] = avg

df = (pd.DataFrame(loss_all).T).sort_index()
model_path = '/data0/BME_caoanbo/project_coding/K-trans_STNet_version1/ST/exp_3_physic_v6/0830_ST_OP_NM_CAhigh_1_2_T8V2_MAE_lr0.0001_batch512_InRawHighDCE_GtRaw_KtransTh_Adam_10subjects_5folds_T10norm_Aug'
df.to_excel(os.path.join(model_path, 'test_result_20_Tmax_vpenumbra6.xlsx'))
print(df)
