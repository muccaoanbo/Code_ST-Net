import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
import argparse
#from model.eTofts import fit_eTofts, full_eTofts, s
import random
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
import pdb

# 获取各种参数
parser = argparse.ArgumentParser()
#parser.add_argument('--input', required=True, type=str)
parser.add_argument('--save_path', required=True, type=str)
parser.add_argument('--output', required=True, choices=['OP_NM_CAlow', 'OP_NM_CAhigh', 'WB_CAhigh', 'WB_CAlow'], type=str)
parser.add_argument('--GT', default='High', choices=['High', 'Low'], type=str)
parser.add_argument('--ratio', nargs='+', default=[1,1], type=int)
parser.add_argument('--patch_steps', default=1, type=int)

args = parser.parse_args()
#pats = args.input
save_path = args.save_path   # ST_pad_InRaw_GtRaw_CAlow
output = args.output
GT = args.GT
ratio = args.ratio
Input = save_path.split('In')[1].split('_')[0]
Output = save_path.split('Gt')[1][:-1].split('_')[0]    #ST_pad_InSmoothed_GtRaw
CA_type = output.split('CA')[1] # 造影剂的浓度high 或者 low
patch_steps = args.patch_steps
print('Input type: ', Input)
print('Output type: ', Output)
#OP_ratio = 6
#NM_ratio = 4
OP_ratio = ratio[0] # opening area
NM_ratio = ratio[1] # normal region
ratio = NM_ratio/OP_ratio
print(ratio)

print('output: {}, GT: {}'.format(output, GT) )

totalvoxel_count = 0 # 定义的全局变量，用来记录总的体素点的个数
def WB_npy(cp, dce_data, T1, brain, ktrans, OP_mask): # 全脑
    print('whole brain')
    WB_datas = []
    count = 1
    for idx in range(OP_mask.shape[2]):
        # print('当前slice：', idx)
        for i in range(OP_mask.shape[0]):
            for j in range(OP_mask.shape[1]):
                # if count % 10000 == 0:
                #     print('已经处理的体素点的个数为：', count / 10000)
                if brain[i, j, idx] == 1:    

                    # 不管是全脑区域还是局部区域，需要将原始图像裁剪成一样的shape
                    raw_dce = dce_data[i-3:i+4, j-3:j+4, idx, :]
                    dce = raw_dce

                    # ktrans的值是存三维的数据，还是只存放单个体素点对应的值，存3维读写太慢
                    # data = {'T10': T1, 'dce_data': dce_data, 'position': (i, j, idx), 'ktrans': ktrans}
                    # data中存放mask中体素点的信号强度值，dce原始data，体素点的位置坐标position，体素点的ktrans值

                    # # 单参数模型
                    # data = {'T10': T1[i, j, idx], 'dce_data': dce, 'position': (i, j, idx), 'ktrans': ktrans[i, j, idx]}

                    # 3参数模型
                    # param合并成一个元组
                    param = np.array([ktrans[0, i, j, idx], ktrans[1, i, j, idx], ktrans[2, i, j, idx]]) # 依次为CBV CBF Tmax
                    data = {'T10': T1[i, j, idx], 'dce_data': raw_dce, 'position': (i, j, idx), 'param': param} # 这里的ktrans是一个4维的数组

                    # x = pool.apply_async(update_data, args=(data, cp, queue)) # 异步执行
                    #data = queue.get()
                    # 不用进程处理
                    # data = update_data(data, cp)  ### 3参数模型时去掉这条语句
                    WB_datas.append(data)
                # count += 1
    WB_count = len(WB_datas)
    global totalvoxel_count
    totalvoxel_count += WB_count
    print(pat, 'WB', WB_count)
    np.save(os.path.join(save_dir, 'WB_{}'.format(WB_count)), WB_datas) # 只将脑组织区域的相关数据存储起来
    print('has saved')
    #return WB_datas, WB_count

def OP_NM_npy(cp, dce_data, T1, brain, ktrans, OP_mask, patch_steps):
    OP_datas = []
    NM_datas = []
    NM_coor = []
    count = 1
    for idx in range(OP_mask.shape[2]):
        for i in range(0, OP_mask.shape[0], patch_steps): # patch_steps表示了裁剪的块之间的重合度
            for j in range(0, OP_mask.shape[1], patch_steps):
                # if count % 10000 == 0:
                #     print('已经处理的体素点的个数为：', count / 10000)
                if OP_mask[i, j, idx] == 1 and brain[i, j, idx] == 1: # 既是脑组织，又是半暗带区域
                    raw_dce = dce_data[i-3:i+4, j-3:j+4, idx, :] # 裁剪成7 * 7 * 84的patches
                    #dce = raw_dce
                    
#                    data = {'T10': T1, 'dce_data': dce_data, 'position': (i, j, idx), 'ktrans': ktrans}

                    # # 单参数模型
                    # data = {'T10': T1[i, j, idx], 'dce_data': raw_dce, 'position': (i, j, idx), 'ktrans': ktrans[i, j, idx]}

                    # 3参数模型
                    # param合并成一个元组
                    param = np.array([ktrans[0, i, j, idx], ktrans[1, i, j, idx], ktrans[2, i, j, idx]]) # 依次为CBV CBF Tmax
                    data = {'T10': T1[i, j, idx], 'dce_data': raw_dce, 'position': (i, j, idx), 'param': param} # 这里的ktrans是一个4维的数组

                    # x = pool.apply_async(update_data, args=(data, cp, queue))

                    # data = queue.get()
                    # data = update_data(data, cp) #### 3参数模型不再需要该语句

                    OP_datas.append(data)

                elif OP_mask[i, j, idx] == 0 and brain[i, j, idx] == 1: # 是脑组织但不是opening area
                    NM_coor.append([i,j,idx]) # 记录下对应的索引
                # count += 1
    OP_count = len(OP_datas)  
    NM_count = int(OP_count*ratio)

    if NM_count > len(NM_coor): # 如果non-opening区域的体素多于opening区域的体素
        NM_count = len(NM_coor)

    # 正常脑组织区域远比半暗带区域要大，这里设置ratio，控制正常组织取的体素点的个数
    normal_datas_random = random.sample(NM_coor, NM_count) # 从一个序列中抽取指定数量的元素，这里相当于挑选出non-opening area的体素
    for X,Y,Z in normal_datas_random:
        raw_dce = dce_data[X-3:X+4, Y-3:Y+4, Z, :] # 裁剪成7 * 7 * 84的形状
        #dce = raw_dce
#        data = {'T10': T1, 'dce_data': dce_data, 'position': (X, Y, Z), 'ktrans': ktrans}
#         # 单参数模型
#         data = {'T10': T1[X, Y, Z], 'dce_data': raw_dce, 'position': (X, Y, Z), 'ktrans': ktrans[X, Y, Z]}

        # 3参数模型
        param = np.array([ktrans[0, X, Y, Z], ktrans[1, X, Y, Z], ktrans[2, X, Y, Z]]) # 依次为CBV CBF Tmax
        data = {'T10': T1[X, Y, Z], 'dce_data': raw_dce, 'position': (X, Y, Z), 'param': param} # 这里的ktrans是一个4维的数组
        # x = pool.apply_async(update_data, args=(data, cp, queue))

        # data = queue.get()
        # data = update_data(data, cp)   # 3参数模型训练时去掉这条语句
        NM_datas.append(data)
           
    print(pat, 'OP:', OP_count, ', NM:', len(NM_coor), ' -> ', len(NM_datas))
    np.save(os.path.join(save_dir, 'OP_{}'.format(OP_count)), OP_datas) # 存储
    np.save(os.path.join(save_dir, 'NM_{}'.format(NM_count)), NM_datas)

    global totalvoxel_count
    totalvoxel_count += (OP_count + NM_count)
    #return OP_datas, NM_datas, OP_count, NM_count




def update_data(data,cp): # 去除q队列参数
    T10, signal, ktrans = data['T10'], data['dce_data'], data['ktrans']		#original
    # 三个参数一起训练时，即保存的就是CBV CBF Tmax
    par = np.array([ktrans, ktrans, ktrans])	#add # 该语句将ktrans值复制了3遍，但值相同
    data.update({'param': par})
    data.update({'dce_data': signal})
    return data
    # q.put(data) # 数据放入队列中

# 代码运行时速度太慢，改用循环处理
# pool = multiprocessing.Pool()
# queue = multiprocessing.Manager().Queue(1024)

# 路径需要更改
# data_root = '/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/' # 原始数据存储路径
data_root = '/data0/BME_caoanbo/project_coding/K-trans_STNet/data/'
# project_root = '/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/DL_input/' # 处理后的数据作为模型输入的存储路径
project_root = '/data0/BME_caoanbo/project_coding/K-trans_STNet/data/DL_input/' # 处理后的数据作为模型输入的存储路径
# project_root = '/data1/BME_caoanbo/project_coding/K-trans_STNet/data/DL_input' # data0磁盘满了，(仅)测试数据放data1

# penumbra对应的id，用来生成所有的测试集，主要是没有半暗带标签的数据
penus_id = []
for penu in os.listdir(os.path.join(data_root, 'raw/openingMask_High')): # 这部分数据用于训练集和验证集，和测试集不重合
    pat = penu[0:15]
    penus_id.append(pat)

sub_count = 0 # 显示程序执行的进度
# for pat_name in os.listdir(os.path.join(data_root, 'raw/openingMask_High')): #
for pat_name in os.listdir(os.path.join(data_root, 'raw/params_High')):
    # pat = pat_name[44:49] # ID号
    pat = pat_name[0:15]
    if pat in penus_id: # 包含在训练集中的数据不在生成测试集
        continue
    if output.split('_CA')[0] == 'OP_NM':
        save_dir = os.path.join(project_root, save_path, '{}_{}_{}_k{}'.format(output, OP_ratio, NM_ratio, GT), '{}'.format(pat))
    else:
        save_dir = os.path.join(project_root, save_path, '{}_k{}'.format(output, GT), '{}'.format(pat))

    os.makedirs(save_dir, exist_ok=True)
    if CA_type == 'low': # 加载raw处理后的muscle中的造影剂浓度的数据
        cp = scio.loadmat(os.path.join(data_root, 'raw_process/Aif_CA', pat, pat+'_CA.mat'))['aif_CA'][:, 0]
    else:
        cp = scio.loadmat(os.path.join(data_root, 'raw_process/Aif_CA', pat, pat+'_CA.mat'))['aif_CA'][:, 0]

    np.save(os.path.join(save_dir, 'CA.npy'), cp) # 存储为npy格式
    brain = nib.load(os.path.join(data_root, 'raw/BrainMask', pat+'.nii.gz')).get_fdata() # 脑组织mask数据
    # OP_mask = nib.load(os.path.join(data_root, 'raw/openingMask_High', pat + '_penumbra.nii.gz')).get_fdata() # opening area
    # OP_mask = OP_mask * brain # OP_mask表示既是脑组织，又是opening area的区域

    # 这里处理WB时，不用考虑异常区域，即生成测试数据集（基于全脑的脑组织体素）
    # 因为需要读入penumbra标签，但不是所有的subject都有标签，防止程序报错
    OP_mask = brain

    if Input == 'RawLowDCE':   # 输入的是低剂量的造影剂数据
        print('Input raw low dose DCE')
        # 加载低剂量的dce数据
        dce_data = nib.load(os.path.join(data_root, 'raw/DCE_Low_4d', pat+'_LowDose_4d.nii.gz')).get_fdata()
        # 造影剂之前的数据，这里是前10个时间序列的数据
        T1 = nib.load(os.path.join(data_root, 'raw_process/T10_LowDose', pat+'_T10_LowDose.nii.gz')).get_fdata()   
#		T1_pcr99 = np.percentile(T1[np.logical_and(brain == 1 , T1 > 0)], 99)
#		print(T1_pcr99)
#		T1 = T1 / T1_pcr99
    elif Input == 'RawHighDCE': # 高剂量
        print('Input raw high dose DCE')
        dce_data = nib.load(os.path.join(data_root, 'raw_process/DSC_High_4d_th', pat+'.nii.gz')).get_fdata()
        # 原始nii的存储坐标系和参数图不一致
        dce_data = np.fliplr(dce_data)
        # break
        T1 = nib.load(os.path.join(data_root, 'raw_process/T10_HighDose_th', pat+'_T10_HighDose.nii.gz')).get_fdata()

        # # 画图，显示方向是否一致
        # plt.figure()
        # plt.imshow(np.squeeze(dce_data[:,:,7,1]))
        # plt.show()
        # plt.figure()
        # plt.imshow(np.squeeze(T1[:,:,7]))
        # plt.show()
        # plt.figure()
        # plt.imshow(np.squeeze(brain[:,:,7]))
        # plt.show()
        # break
    else:
        print('Wrong input')
        exit()


    if Output == 'Raw': # 加载ktrans数据，加载参数的金标准
        print('Output raw')
        # ktrans = nib.load(os.path.join(data_root, 'raw/params_High', pat, pat+'_rCBV.nii.gz')).get_fdata()
        # ktrans = nib.load(os.path.join(data_root, 'raw/params_High', pat, pat + '_rCBF.nii.gz')).get_fdata()
        # ktrans = nib.load(os.path.join(data_root, 'raw/params_High', pat, pat + '_Tmax.nii.gz')).get_fdata()

        # 同时生成cbv、cbf、Tmax参数时
        CBV = nib.load(os.path.join(data_root, 'raw/params_High', pat, pat+'_rCBV.nii.gz')).get_fdata()
        CBF = nib.load(os.path.join(data_root, 'raw/params_High', pat, pat + '_rCBF.nii.gz')).get_fdata()
        Tmax = nib.load(os.path.join(data_root, 'raw/params_High', pat, pat + '_Tmax.nii.gz')).get_fdata()
        params = np.array([CBV, CBF, Tmax]) # 合并这三个元素，相当于拼接成了一个4维的数组

#		ktrans = nib.load(os.path.join(data_root, 'Raw/Ktrans_High', pat+'_HighDose_ktrans.nii.gz')).get_fdata()
    else:
        print('Wrong output')
        exit()


    T10_WB = T1 * brain # 前10个序列的脑组织区域
    T10_pcr99 = np.percentile(T10_WB[brain == 1], 99) # 计算数据的百分位数
    print(T10_pcr99)
    os.makedirs(os.path.join(project_root, 'T10_prc99'), exist_ok=True)
    np.save(os.path.join(project_root, 'T10_prc99', pat), T10_pcr99)

 #   plt.subplot(2,3,1)
 #   plt.plot(cp)

 #   plt.subplot(2,3,2)
 #   plt.imshow(dce_data[:,:,10,83])

 #   plt.subplot(2,3,3)
	# plt.imshow(T1[:,:,10])

 #   plt.subplot(2,3,4)
 #   plt.imshow(brain[:,:,10])

 #   plt.subplot(2,3,5)
 #   plt.imshow(ktrans[:,:,10])

 #   plt.subplot(2,3,6)
 #   plt.imshow((OP_mask*ktrans)[:,:,10])
	# plt.show()


    if output.split('_CA')[0] == 'WB': # 全脑区域
       # WB_npy(cp, dce_data, T1, brain, ktrans, OP_mask)
       WB_npy(cp, dce_data, T1, brain, params, OP_mask) # 3参数模型
    elif output.split('_CA')[0] == 'OP_NM': # opening area 和 正常区域（即non-opening area）
       # OP_NM_npy(cp, dce_data, T1, brain, ktrans, OP_mask, patch_steps)
       OP_NM_npy(cp, dce_data, T1, brain, params, OP_mask, patch_steps) # 3参数模型
    else:
       exit()

    # 显示目前处理了多少个
    sub_count += 1
    print(sub_count)
print('save_dir: ',save_dir) # save_dir是处理好，作为模型输入的数据
print('全脑的总的体素点是：', totalvoxel_count) # 这里的全脑只计算了有半暗带标签的，72份

#python3 from_mat_to_npy.py --save_path=ST_pad_InRawLowDCE_GtRaw_KtransTh/ --output='OP_NM_CAlow' --ratio 1 1
#python3 from_mat_to_npy.py --save_path=ST_pad_InRawLowDCE_GtRaw_KtransTh_patchsteps3/ --output='OP_NM_CAlow' --ratio 1 4 --patch_steps 3
#python3 from_mat_to_npy.py --save_path=ST_pad_InRawLowDCE_GtRaw_KtransTh/ --output='WB_CAlow' 

#python3 from_mat_to_npy.py --save_path=ST_pad_InRawHighDCE_GtRaw_KtransTh/ --output='OP_NM_CAhigh' --ratio 1 1
#python3 from_mat_to_npy.py --save_path=ST_pad_InRawHighDCE_GtRaw_KtransTh_patchsteps3/ --output='OP_NM_CAhigh' --ratio 1 4 --patch_steps 3
#python3 from_mat_to_npy.py --save_path=ST_pad_InRawHighDCE_GtRaw_KtransTh/ --output='WB_CAhigh' 



#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InSmoothed_GtRaw/ --output='OP_NM' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtSmoothed/ --output='OP_NM' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InSmoothed_GtSmoothed/ --output='OP_NM' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtRaw/ --output='OP_NM' --ratio 1 1

#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InSmoothed_GtRaw/ --output='WB'
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtSmoothed/ --output='WB'
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InSmoothed_GtSmoothed/ --output='WB'
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtRaw/ --output='WB'


#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtMedianTh/ --output='OP_NM' --ratio 1 4
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtRaw_nonoverlap/ --output='OP_NM' --ratio 1 1

#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_Low/ --output='OP_NM_CAhigh' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_Low/ --output='OP_NM_CAlow' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_Low/ --output='WB_CAhigh' 
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_Low/ --output='WB_CAlow' 

#python3 from_mat_to_npy.py --save_path=ST_pad_InRaw_GtRaw_KtransTh/ --output='OP_NM_CAlow' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_KtransTh_T10nor_patchsteps3/ --output='OP_NM_CAlow' --ratio 1 4 --patch_steps 3
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_KtransTh/ --output='WB_CAlow' 

#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_KtransTh_patchsteps3_pos/ --output='OP_NM_CAlow' --ratio 1 4 --patch_steps 3
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_KtransTh_pos/ --output='WB_CAlow' 

