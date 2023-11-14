#
# 用来计算重建出来的参数图和GD之间的指标值
import argparse
import datetime
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# from model.network_model_T import CNN_T
from utils.evaluate import concordance_correlation_coefficient as ccc
from utils.evaluate import KL

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

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
args = parser.parse_args()
model_path = os.path.join('exp5/',args.model_path)
config_path = os.path.join(model_path, 'config.toml')


with open(config_path, 'r', encoding='utf-8') as f:
	config = toml.load(f)
	default_config = EasyDict(config)

fold_df = default_config.train.fold_excel # 指定需要计算的subject
fold_df = pd.read_excel(fold_df, dtype={'subject':str, 'fold':int})
test_config = default_config.test
test_config.update(args.__dict__)

gpu = torch.device('cuda:{}'.format(default_config.train.gpu))
torch.cuda.set_device(gpu)
torch.multiprocessing.set_sharing_strategy('file_system')
batch = test_config.batch

str_time = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))

# fast_m = CNN_T().cuda()
test_path = test_config.dataset
print('test_path',test_path)

loss_type = default_config.train.loss_type
if loss_type == 'MSE':
	loss_func = nn.MSELoss()
elif loss_type == 'MAE':
	loss_func = nn.L1Loss()
elif loss_type == 'SmoothL1':
	loss_func = nn.SmoothL1Loss()

def test(name):

	res_x = []
	res_y = []

	k_trans_x = []
	k_trans_y = []

	test_img = nib.load(os.path.join(model_path, 'test', name)).get_fdata()
	GT_img = nib.load(os.path.join('/data0/BME_caoanbo/project_coding/K-trans_STNet/data/raw/params_High', name[-27:-12],  name[-27:-12] + '_rCBV.nii.gz')).get_fdata()
	# BrainMask = nib.load(os.path.join('/data0/BME_caoanbo/project_coding/K-trans_STNet/data/raw/BrainMask', BrainMask_ID[name[-17:-12]] + '.nii.gz')).get_fdata()
	# 指定评估的区域，这里评估penumbra区域
	BrainMask = nib.load(os.path.join('/data0/BME_caoanbo/project_coding/K-trans_STNet/data/raw/openingMask_High_test', name[-27:-12] + '_penumbra.nii.gz')).get_fdata()

	test_WB = test_img * BrainMask
	GT_WB = GT_img * BrainMask


	# kernel_size = (3,3,3)
	# test_WB_smoothed = ndimage.median_filter(test_WB, size = kernel_size)
	# GT_WB_smoothed = ndimage.median_filter(GT_WB, size = kernel_size)
	# GT_WB_smoothed[GT_WB_smoothed > 0.05] = 0
	test_WB_smoothed = test_WB
	GT_WB_smoothed = GT_WB

	GT_WB_max = np.max(GT_WB_smoothed[BrainMask == 1])
	test_WB_max = np.max(test_WB_smoothed[BrainMask == 1])

	GT_WB_smoothed = GT_WB_smoothed #/ GT_WB_max
	test_WB_smoothed = test_WB_smoothed #/ test_WB_max

	th = 0
	# INCRESE TH TO 0.2 AND DO COMPARISON (for opening area)
	GT_WB_smoothed[GT_WB_smoothed < th] = 0
	# GT_WB_smoothed[GT_WB_smoothed > 1] = 0
	test_WB_smoothed[test_WB_smoothed < th] = 0
	# test_WB_smoothed[test_WB_smoothed > 1] = 0


	GT_WB_smoothed_array = GT_WB_smoothed[BrainMask == 1]
	test_WB_smoothed_array = test_WB_smoothed[BrainMask == 1]


	Prediction_vs_GT_SCC, p_value = scipy.stats.spearmanr(test_WB_smoothed_array, GT_WB_smoothed_array, nan_policy='omit')
	Prediction_vs_GT_PCC, p_value = np.corrcoef(GT_WB_smoothed_array, test_WB_smoothed_array)[1]
	Prediction_vs_GT_ccc = ccc(GT_WB_smoothed_array, test_WB_smoothed_array)
	Prediction_vs_GT_nrmse = nrmse(GT_WB_smoothed_array, test_WB_smoothed_array, normalization='euclidean')
	Prediction_vs_GT_KL = KL(test_WB_smoothed_array, GT_WB_smoothed_array, BrainMask, 128)

	Prediction_vs_GT_PSNR_sum = 0
	Prediction_vs_GT_SSIM_sum = 0
	slice_count = 0
	for i in range(test_img.shape[2]):
		GT_WB_smoothed_slice = GT_WB_smoothed[:,:,i]
		test_WB_smoothed_slice = test_WB_smoothed[:,:,i]
		BrainMask_slice = BrainMask[:, :, i]

		# 先对切片中的信号强度值做归一化，因为psnr和ssim计算的时候对值的范围归一化到了-1到1
		# 为防止分母为0，需要平滑处理
		# print(GT_WB_smoothed_slice.size)
		GT_WB_smoothed_slice = (GT_WB_smoothed_slice - np.min(GT_WB_smoothed_slice)) / ((np.max(GT_WB_smoothed_slice) - np.min(GT_WB_smoothed_slice)) + 1e-6)
		test_WB_smoothed_slice = (test_WB_smoothed_slice - np.min(test_WB_smoothed_slice)) / ((np.max(test_WB_smoothed_slice) - np.min(test_WB_smoothed_slice)) + 1e-6)

		# 原始的mask中在某些slice上没有对应的脑组织，不存在脑组织区域
		if len(GT_WB_smoothed_slice[BrainMask_slice == 1]) == 0 or len(test_WB_smoothed_slice[BrainMask_slice == 1]) == 0:
			continue
		slice_psnr = PSNR(GT_WB_smoothed_slice[BrainMask_slice == 1], test_WB_smoothed_slice[BrainMask_slice == 1])
		try:
			slice_ssim = SSIM(GT_WB_smoothed_slice[BrainMask_slice == 1], test_WB_smoothed_slice[BrainMask_slice == 1])
		except Exception as e:
			print(name[-27:-12]) # 打印计算报错的subject
		Prediction_vs_GT_PSNR_sum += slice_psnr
		Prediction_vs_GT_SSIM_sum += slice_ssim

		slice_count += 1
	Prediction_vs_GT_PSNR = Prediction_vs_GT_PSNR_sum / slice_count
	Prediction_vs_GT_SSIM = Prediction_vs_GT_SSIM_sum / slice_count

	# Prediction_vs_GT_SSIM = SSIM(GT_WB_smoothed_array, test_WB_smoothed_array)
	# Prediction_vs_GT_SSIM = 0
	# for i in range(18):
	# 	t = test_WB_smoothed[i]
	# 	g = GT_WB_smoothed[i]
	# 	b = BrainMask[i]
	# 	Prediction_vs_GT_SSIM = SSIM(t[b==1], g[b==1])#, datarange=test_WB_smoothed_array.max()-test_WB_smoothed_array.min())
	# 	print(Prediction_vs_GT_SSIM)

	loss_dict = {'SCC': Prediction_vs_GT_SCC, 'PCC': Prediction_vs_GT_PCC, 'CCC': Prediction_vs_GT_ccc, 'NRMSE': Prediction_vs_GT_nrmse, 'KL': Prediction_vs_GT_KL, 'PSNR': Prediction_vs_GT_PSNR, 'SSIM': Prediction_vs_GT_SSIM}
	loss_info = '{}, SCC:{}, PCC:{}, CCC:{}, NRMSE:{}, KL:{}, PSNR:{}, SSIM:{}'.format(
				name, Prediction_vs_GT_SCC, Prediction_vs_GT_PCC, Prediction_vs_GT_ccc, Prediction_vs_GT_nrmse, Prediction_vs_GT_KL, Prediction_vs_GT_PSNR, Prediction_vs_GT_SSIM
	)

	return loss_info, loss_dict

if not os.path.exists(model_path):
	raise FileNotFoundError(f'{model_path} not exists!')

test_time = str(datetime.datetime.now())
# dict1 = {'SCC': 0, 'PCC': 0, 'CCC': 0, 'NRMSE': 0, 'KL': 0, 'PSNR': 0, 'SSIM': 0}
loss_all = {}

for f in range(1, len(set(fold_df['fold']))+1):
	testing_subject = fold_df.loc[fold_df['fold'] == f, ['subject']]['subject'].tolist() # == f

	# test_result = []

	for pat_dir in testing_subject: # 此处的pat_dir即为subject_id
		save_name = 'Fold' + str(f) + '_' + pat_dir + '_test.nii.gz' # 存储的预测的ktrans.nii.gz数据
		if not os.path.exists(os.path.join(model_path, 'test', save_name)):
			continue

		pat_result, loss_dict = test(save_name) # 这里只计算指标值进行存储即可，test.py已经生成了重建后的nii.gz
		loss_all[pat_dir] = loss_dict # lossdict是一个字典类型的数据，存储的是各参数的值

		# test_result.append(pat_result)
		# for metrics, val in dict1.items():
		# 	val += loss_dict[metrics]
		# 	dict1[metrics] = val

print('------------Test Result------------')


# df_loss_all = pd.DataFrame(loss_all).T
# for metrics, val in dict1.items():
# 	val /= len(df_loss_all)
# 	dict1[metrics] = val
# dict1 = {'Average': dict1}
# df_dict1 = pd.DataFrame(dict1).T
# df = pd.concat([df_loss_all, df_dict1])
# print(df)
df = pd.DataFrame(loss_all).T # dataframe转置
loss_all['Avg'] = {} # 用来记录每个参数的均值
for k, v in df.items():
    avg = v.mean()
    loss_all['Avg'][k] = avg

df = (pd.DataFrame(loss_all).T).sort_index() # 转置回去
df.to_excel(os.path.join(model_path, 'test_result_20_CBV_penumbra .xlsx'))
print(df)
print('Test start at', test_time)
print('End at', str(datetime.datetime.now()), f'result path:{model_path}')
