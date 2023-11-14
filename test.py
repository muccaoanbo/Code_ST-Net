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

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
args = parser.parse_args()
model_path = os.path.join('exp_3_physic_v6/',args.model_path)
config_path = os.path.join(model_path, 'config.toml') # config.toml有测试的参数设置


with open(config_path, 'r', encoding='utf-8') as f:
	config = toml.load(f)
	default_config = EasyDict(config)

fold_df = default_config.train.fold_excel
fold_df = pd.read_excel(fold_df, dtype={'subject':str, 'fold':int})
test_config = default_config.test
test_config.update(args.__dict__)

gpu = torch.device('cuda:{}'.format(default_config.train.gpu))
# gpu = torch.device('cuda:6')
torch.cuda.set_device(gpu)
torch.multiprocessing.set_sharing_strategy('file_system')
batch = test_config.batch # 测试时的batch设置

str_time = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))

fast_m = CNN_ST().cuda() # 加载模型
test_path = test_config.dataset # 加载测试data
print('test_path',test_path)

loss_type = default_config.train.loss_type # 损失函数的设置
if loss_type == 'MSE':
	loss_func = nn.MSELoss()
elif loss_type == 'MAE':
	loss_func = nn.L1Loss()
elif loss_type == 'SmoothL1':
	loss_func = nn.SmoothL1Loss()

def test(dataset, fast_m, name='test'):

	res_x = []
	res_y = []

	k_trans_x = []
	k_trans_y = []

	with torch.no_grad():
		fast_m.eval()
		test_load = DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=16) # num_workers原来为4

		# 有CBV CBF Tmax三个参数的金标准
		# img_name = name + '_rCBV.nii.gz' # 此处的name应该是传入的参数为ID
		img_name = name + '_rCBF.nii.gz'  # 此处的name应该是传入的参数为ID
		# img_name = name + '_Tmax.nii.gz'  # 此处的name应该是传入的参数为ID
		# img是真实的ktrans参数图，是三维的，且未做平滑处理的
		img = nib.load(os.path.join('/data0/BME_caoanbo/project_coding/K-trans_STNet/data/raw/params_High', name, img_name))
		test_img = np.zeros(img.shape) # 模型预测得到的值
		GT_img = np.zeros(img.shape) # 金标准的值

		for count, data in enumerate(test_load):
			dce_spetial, data_ref, paramters, pos = data   # pos表示的是体素点的坐标，用来重建完整的参数图
			dce_spetial = dce_spetial.cuda()
			# print(dce_spetial.shape)
			data_ref = data_ref.cuda()
			# print(data_ref.shape)

			# try:
			pre = fast_m(dce_spetial, data_ref) # 这里的data_ref是新增的两个特征，所以data_ref应该是2 * 84
			# except Exception as e:
			# 	print(batch)
			# 	print(count)
			# 	print(count * batch)
			# 	print(dce_spetial.shape)
			# 	print(data_ref.shape)

			# 训练时对参数做了归一化
			# pre[:, 0] = pre[:, 0] * 129.86 # CBV参数
			pre[:, 1] = pre[:, 1] * 655.35 # CBF参数
			# pre[:, 2] = pre[:, 2] * 56 # Tmax参数
			# pre[:, 0] = pre[:, 0] * 5  # CBV参数
			# 单个测试时需要指定参数得索引 CBV：0 CBF: 1 Tmax: 2
			test_img[pos[0].numpy(), pos[1].numpy(), pos[2].numpy()] = pre[:, 1].cpu().numpy() # 坐标点对应的值（整个批次一起处理）
			GT_img[pos[0].numpy(), pos[1].numpy(), pos[2].numpy()] = paramters[:, 1].cpu().numpy() # 金标准对应的值

	# 预测的cbv的值如果小于0，将其置为0
	test_img[test_img < 0] = 0
	PredictionImg = nib.Nifti1Image(test_img, img.affine, img.header) # 存储成3d.nii.gz数据时对应的格式
	
	# 读入brain mask
	# BrainMask = nib.load(os.path.join('/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/raw/BrainMask', BrainMask_ID[name] + '.nii.gz')).get_fdata()
	BrainMask = nib.load(os.path.join('/data0/BME_caoanbo/project_coding/K-trans_STNet/data/raw/BrainMask', name + '.nii.gz')).get_fdata()
	test_WB = test_img * BrainMask # 乘以mask，只考虑脑组织区域的
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

	GT_WB_smoothed = GT_WB_smoothed #/ GT_WB_max
	test_WB_smoothed = test_WB_smoothed #/ test_WB_max

	th = 0
	# INCRESE TH TO 0.2 AND DO COMPARISON (for opening area)
	GT_WB_smoothed[GT_WB_smoothed < th] = 0 # cbv的值不可能小于0
	# GT_WB_smoothed[GT_WB_smoothed > 100] = 0
	test_WB_smoothed[test_WB_smoothed < th] = 0
	# test_WB_smoothed[test_WB_smoothed > 100] = 0


	GT_WB_smoothed_array = GT_WB_smoothed[BrainMask == 1] # 只考虑brain mask区域，这时变成了一维数组
	test_WB_smoothed_array = test_WB_smoothed[BrainMask == 1]

	# 计算评估参数
	Prediction_vs_GT_SCC, p_value = scipy.stats.spearmanr(test_WB_smoothed_array, GT_WB_smoothed_array, nan_policy='omit')
	Prediction_vs_GT_PCC, p_value = np.corrcoef(GT_WB_smoothed_array, test_WB_smoothed_array)[1]
	Prediction_vs_GT_ccc = ccc(GT_WB_smoothed_array, test_WB_smoothed_array)
	Prediction_vs_GT_nrmse = nrmse(GT_WB_smoothed_array, test_WB_smoothed_array, normalization='euclidean')
	Prediction_vs_GT_KL = KL(test_WB_smoothed_array, GT_WB_smoothed_array, BrainMask, 128) # KL散度

	Prediction_vs_GT_PSNR_sum = 0
	Prediction_vs_GT_SSIM_sum = 0
	slice_count = 0
	for i in range(test_img.shape[2]): # slice维度上
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
		slice_ssim = SSIM(GT_WB_smoothed_slice[BrainMask_slice == 1], test_WB_smoothed_slice[BrainMask_slice == 1])

		Prediction_vs_GT_PSNR_sum += slice_psnr
		Prediction_vs_GT_SSIM_sum += slice_ssim
		slice_count += 1 # 记录多少个slice参与了psnr和ssim的计算，因为有些slice上没有脑组织，不用考虑进来
	Prediction_vs_GT_PSNR = Prediction_vs_GT_PSNR_sum / slice_count # 平均值
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

	return loss_info, PredictionImg, loss_dict

if not os.path.exists(model_path):
	raise FileNotFoundError(f'{model_path} not exists!')

test_time = str(datetime.datetime.now())
# dict1 = {'SCC': 0, 'PCC': 0, 'CCC': 0, 'NRMSE': 0, 'KL': 0, 'PSNR': 0, 'SSIM': 0}
loss_all = {}

for f in range(1, len(set(fold_df['fold']))+1):
	testing_subject = fold_df.loc[fold_df['fold'] == f, ['subject']]['subject'].tolist() # 这里需要重新指定测试的id，这里为3
	if not os.path.exists(os.path.join(model_path, 'Fold' + str(f), 'best_patient.tar')): # str中的参数是f
		continue

	param_path = os.path.join(model_path, 'Fold' + str(f), 'best_patient.tar') # str中的应该是参数f
	best_model = torch.load(param_path, map_location=gpu)
	fast_m.load_state_dict(best_model) # 加载模型

	# test_result = []
	
	save_path = os.path.join(model_path,'test')
	if not os.path.exists(save_path):
		os.makedirs(save_path) # 生成测试生成数据的存储路径
	# print('testing subjects',testing_subject)
	sub_count = 0
	for pat_dir in testing_subject:
		# print(test_path)
		test_pat_data = Patient(os.path.join(test_path, pat_dir)) # 测试数据集
		pat_result, PredictionImg, loss_dict = test(test_pat_data, fast_m, pat_dir)
		loss_all[pat_dir] = loss_dict
		save_name = 'Fold' + str(f) + '_' + pat_dir + '_test_CBF.nii.gz' # str中的参数是f
		nib.save(PredictionImg, os.path.join(save_path, save_name)) # 保存数据

		sub_count += 1
		print(sub_count)

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
# 存储测试时计算的评估指标的值
df = pd.DataFrame(loss_all).T
loss_all['Avg'] = {}
for k, v in df.items():
    avg = v.mean()
    loss_all['Avg'][k] = avg

df = (pd.DataFrame(loss_all).T).sort_index()
df.to_excel(os.path.join(model_path, 'test_result_86_CBF.xlsx'))
print(df)
print('Test start at', test_time)
print('End at', str(datetime.datetime.now()), f'result path:{model_path}')
