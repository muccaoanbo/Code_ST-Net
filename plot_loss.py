import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
#from utils.config import get_config
import numpy as np
import toml
from easydict import EasyDict

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
args = parser.parse_args()
model_path = args.model_path # 训练好的模型相关文件存储的位置
if os.path.isfile(os.path.join('exp4', model_path, 'config_loadCKPT.toml')): # 检查某个文件是否存在
    config_path = os.path.join('exp4', model_path, 'config_loadCKPT.toml')
elif os.path.isfile(os.path.join('exp4', model_path, 'config.toml')):
    config_path = os.path.join('exp4', model_path, 'config.toml') # config.toml是训练时的各种参数的设置

with open(config_path, 'r', encoding='utf-8') as f:
    config = toml.load(f)
    default_config = EasyDict(config)
fold_df = default_config.train.fold_excel
fold_df = pd.read_excel(fold_df, dtype={'subject':str, 'fold':int}) # 交叉验证的excel表

plt.figure()
r, c = 2, len(set(fold_df['fold'])) # c表示的是分成了几折验证
for i in range(1, len(set(fold_df['fold'])) + 1): # 绘制每一次交叉验证的结果
	# if os.path.isfile(os.path.join('exp/', model_path, 'models/Fold'+str(i), 'learning_curve.csv')):
	# 	train_loss = pd.read_csv(os.path.join('exp/',model_path,'models/Fold'+str(i), 'learning_curve.csv'))
	if os.path.isfile(os.path.join('exp4/', model_path, 'Fold'+str(i), 'learning_curve.csv')): # 读入训练时存储好的各参数
		train_loss = pd.read_csv(os.path.join('exp4/',model_path,'Fold'+str(i), 'learning_curve.csv'))
		Pear_train_avg_list = train_loss['Train Pearson Corr'].tolist()
		Pear_val_avg_list = train_loss['Val Pearson Corr'].tolist()
		# Pear_test_avg_list = train_loss['Test Pearson Corr'].tolist()
		loss_type_val = train_loss.columns[2] # 取出各损失对应的列的索引
		loss_type_train = train_loss.columns[1]
		# loss_type_Test = train_loss.columns[3]

		Err_val_avg_list = train_loss[loss_type_val].tolist() # 取出各损失的值
		Err_train_avg_list = train_loss[loss_type_train].tolist()
		# Err_Test_avg_list = train_loss[loss_type_Test].tolist()

		best_epoch = train_loss[loss_type_val].idxmin() # 验证时的最小损失对应的epoch为最好的epoch
		#best_epoch = len(Err_val_avg_list)-1001
		print('epoch: {}, MSE(T): {}, PR(T): {}, MSE(V): {}, PR(V): {}'.format(best_epoch, Err_train_avg_list[best_epoch],Pear_train_avg_list[best_epoch], Err_val_avg_list[best_epoch], Pear_val_avg_list[best_epoch]))


		
		plt.subplot(r, c, i) # 这里的i表示对应的哪一折
		plt.plot(Err_train_avg_list,'r') # 训练红线
		plt.plot(Err_val_avg_list,'b') # 验证蓝线
		# plt.plot(Err_Test_avg_list,'g') # 测试绿线
		plt.axvline(best_epoch, color='y') # 标记出最好的epoch的位置
		plt.legend(['Train','Validation', 'Test', 'Best Epoch='+str(best_epoch)])
		plt.title('Fold'+ str(i)+ ' '+ 'Loss')

		plt.subplot(r, c, i + c) # pearson系数训练、验证和测试的变化
		plt.plot(Pear_train_avg_list,'r')
		plt.plot(Pear_val_avg_list,'b')
		# plt.plot(Pear_test_avg_list,'g')
		plt.axvline(best_epoch, color='y')
		plt.legend(['Train','Validation', 'Test', 'Best Epoch='+str(best_epoch)])
		plt.title('Pearson Correlation')
		plt.suptitle(model_path.split('/')[-1]) # 添加一个总标题
plt.show()
