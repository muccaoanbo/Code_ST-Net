# 统计rapid中的参数值的范围
import nibabel as nb
import os
import matplotlib.pyplot as plt
import numpy as np


def threshold(path, name): # 确定出所有subject中参数对应的最大值和最小值
    subs = os.listdir(path)
    max_value = 10000
    for sub in subs:
        param_path = os.path.join(path, sub, sub + name + '.nii.gz')
        img = nb.load(param_path).get_fdata()
        # max = img.max()
        min = img.min()
        print(sub, min)
        # if max > max_value:
        #     max_value = max
        if min < max_value:
            max_value = min
    return max_value

def draw_hist(path, name): # 绘制出参数值分布的直方图
    subs = os.listdir(path)
    all_fdata = []
    result = np.array([])
    for sub in subs:
        param_path = os.path.join(path, sub, sub + name + '.nii.gz')
        img = nb.load(param_path).get_fdata() # 读取nii.gz array
        img_flat = img.flatten()
        # all_fdata.append(img_flat)
        result = np.concatenate((result, img_flat))
    hist, bin_edges = np.histogram(result, bins=100)
    print(hist)
    print(bin_edges)
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # # 绘制合并后的灰度值数据的直方图
    # plt.hist(all_fdata, bins=50, alpha=0.7)
    # plt.title('CBV影像灰度值直方图')
    # plt.xlabel('灰度值')
    # plt.ylabel('频率')
    # plt.grid(True)
    # # 显示直方图
    # plt.show()


params_path = '/data0/BME_caoanbo/project_coding/K-trans_STNet/data/raw/params_High'
name = '_Tmax'
draw_hist(params_path, name)
