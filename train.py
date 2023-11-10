import argparse
import datetime
import os
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from utils.dataset_ST import patients_dataset, Patient, dataAug
from model.network_model_ST import CNN_ST, perfusion
from utils.config import get_config

import csv
from shutil import copyfile
import pandas as pd
import torchio as tio
import glob

x1 = 22.7
TE = 32

default_config = get_config() # 默认读取config.toml文件
train_config = default_config.train # 训练的参数设置

# apply config
dataset =           train_config['dataset']
train_input =       dataset.split('/')[-1].split('_k')[0]
data_type =         dataset.split('/')[-2].split('pad_')[1]
curr_fold =         train_config.curr_fold
batch =             train_config.batch
lr =                train_config.lr
loss_type =         train_config.loss_type
ckpt_path =         train_config.ckpt_path
Optimizer_type =    train_config.Optimizer_type
fold_df =           train_config.fold_excel # 较差验证设置的excel文件
stop_num =          train_config.stop_num
max_epoch =         train_config.max_epoch
train_val_ratio =   train_config.train_val_ratio
Augmentation =      train_config.Augmentation

gpu =               torch.device('cuda:{}'.format(train_config.gpu)) # 选择的GPU
torch.cuda.set_device(gpu)
torch.multiprocessing.set_sharing_strategy('file_system')

str_time = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
print('Training fold: ', curr_fold) # 训练折
print(fold_df)
model_path = os.path.join('./exp_3_physic_v6/0830_ST_' + train_input + '_T{0}V{1}_{2}_lr{3}_batch{4}_{5}_{6}_10subjects_5folds_T10norm_Aug'.format(
                               int(train_val_ratio*10), 10-int(train_val_ratio*10), loss_type, lr, batch, data_type, Optimizer_type))

# 设置损失函数
if loss_type == 'MSE':
    loss_func = nn.MSELoss()
elif loss_type == 'MAE':
    loss_func = nn.L1Loss()
elif loss_type == 'SmoothL1':
    loss_func = nn.SmoothL1Loss()

print('model save to:', model_path.split('/')[-1])


def train(train_set, valid_set, test_set, fast_m, perfusion_m, optimizer, fold, augmentation, name='train'):
    '''
    :param train_set:
    :param valid_set:
    :param fast_m:
    :param optimizer:
    :param name:
    :return:
    '''
    copyfile('config.toml', model_path + '/config.toml')
    copyfile('./model/network_model_ST.py', model_path + '/network_model_ST.py')
    copyfile('./utils/dataset_ST.py', model_path + '/dataset_ST.py')
    min_loss = torch.FloatTensor([float('inf'), ])
    #max_pear = torch.FloatTensor([0, ])

    global stop_num
    ealy_stop = stop_num

    #################### Learning Curve ####################
    LossPath = model_path  + '/Fold' + fold

    with open(LossPath + '/learning_curve.csv', 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        # 训练集、验证集、测试集
        # wr.writerow(['Current Epoch', 'Train {} Loss'.format(loss_type), 'Val {} Loss'.format(loss_type), 'Test {} Loss'.format(loss_type),
        #             'Train Pearson Corr', 'Val Pearson Corr', 'Test Pearson Corr'])

        # 训练和验证集
        wr.writerow(['Current Epoch', 'Train {} Loss'.format(loss_type), 'Val {} Loss'.format(loss_type),
                    'Train Pearson Corr', 'Val Pearson Corr'])
        f.close()
    #################### Learning Curve ####################

    for epoch in range(max_epoch):
        loss_array_train,  loss_array_val, loss_array_test= [], [], []           #add_0901
        pearson_corr_train, pearson_corr_val, pearson_corr_test = [], [], []       #add_0901

########################################################## Train ##########################################################

        dataloader = DataLoader(dataset=train_set, batch_size=batch, shuffle=True, num_workers=16)

        fast_m.train()
        for count, data in enumerate(dataloader): 
            # 此处的data_ref一个是动脉输入函数，一个是没有造影剂的基线信号强度，data_ref中可以分离出T10
            dce_spatial, data_ref, paramters, pos = data # load dce, ktrans, 4d, muscle dce
            # print(type(paramters))
            # print(paramters.shape)
            # print(dce_spatial[0,0,3,3,:]) # 在裁块中找到原始的体素点对应的信号强度的变化，该信号强度值值经过归一化处理
            # print(data_ref[0, 0, :])
            # print(data_ref[0, 1, :])
            if augmentation:
                dce_spatial = dataAug(dce_spatial)
    		
            dce_spatial = dce_spatial.cuda()
            data_ref = data_ref.cuda()

            # 3参数模型
            # 归一化，保证对损失函数的贡献一致
            paramters[:, 0] = paramters[:, 0] / 129.86
            paramters[:, 1] = paramters[:, 1] / 655.35
            paramters[:, 2] = paramters[:, 2] / 56.0

            paramters = paramters.cuda()

            pre = fast_m(dce_spatial, data_ref) #CNN predict output, 输出的形状为(batch_size, 1) 回归出一个值

            # ## 单参数模型
            # paramters[:, 0] = paramters[:, 0]  # 归一化cbv损失，协调两个损失函数得作用
            # paramters= paramters[:, 0] # 512表示批量，第二个维度表示的是对应的ktrans参数值
            # # 设置阈值，将参数值控制在某个范围内
            # # paramters_th = paramters[paramters<=0.05]
            # # pre_th = pre[paramters<=0.05]
            # paramters_th = paramters
            # pre_th = pre
            # pre_loss = loss_func(pre_th[:,0], paramters_th)    # pre -> CNN predict results; parameters(obtained from NLLS fitting) -> ground truth

            # 从原始数据计算时间浓度函数的积分
            raw_s = dce_spatial[:,0,3,3,:]
            step_size = 1 # 时间步长
            # ct_t = x1 * (-1 / TE) * torch.log(raw_s / data_ref[:, 1, :])
            ct_t = (-1 / TE) * torch.log(raw_s / data_ref[:, 1, :])  # 去掉x1，进一步简化模型
            ct_t_integral = torch.zeros(raw_s.size(0), 1)
            for t in range(0, raw_s.size(-1)):
                integral = 0.0
                for i in range(1, raw_s.size(-1)):
                    integral += (ct_t[t, i - 1] + ct_t[t, i]) * step_size / 2.0
                ct_t_integral[t, :] = integral
            # print('原始数据', ct_t_integral[0, :])
            ct_t_integral = ct_t_integral.to(paramters.device)

            sig_pre = perfusion_m(pre, data_ref[:, 1, :], data_ref[:, 0, :], raw_s) # 传入的第一个参数应该是模型预测得到的CBV参数
            # print('cbv推导', sig_pre[0, :])

            # loss1 = pre_loss
            # loss2 = loss_func(ct_t_integral, sig_pre) * 20  # loss1 和 loss2相差了一个数量级，赋予loss1和loss2不同的权重
            # # print(loss1, loss2)
            # loss = loss1 + loss2
            # loss_array_train.append(loss.item())        #add_0901，存储损失
            # x = pre_th[:, 0]
            # y = paramters_th
            # vx = x - torch.mean(x)
            # vy = y - torch.mean(y)
            # pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            # pearson_corr_train.append(pearson_corr.item()) # 计算pcc

            ### 3参数模型
            paramters_th = paramters
            pre_th = pre
            pre_loss = loss_func(pre_th,paramters_th)  # pre -> CNN predict results; parameters(obtained from NLLS fitting) -> ground truth

            loss = torch.mean(pre_loss) # 3个参数，求均值
            # 加入物理损失函数
            loss2 = loss_func(ct_t_integral, sig_pre)  # loss1 和 loss2相差了一个数量级
            # print(loss, loss2)
            loss = loss + loss2
            loss_array_train.append(loss.item())  # add_0901，存储损失

            x1 = pre_th[:, 0]
            y1 = paramters_th[:, 0]
            vx1 = x1 - torch.mean(x1)
            vy1 = y1 - torch.mean(y1)
            pearson_corr1 = torch.sum(vx1 * vy1) / (torch.sqrt(torch.sum(vx1 ** 2)) * torch.sqrt(torch.sum(vy1 ** 2)))

            x2 = pre_th[:, 1]
            y2 = paramters_th[:, 1]
            vx2 = x2 - torch.mean(x2)
            vy2 = y2 - torch.mean(y2)
            pearson_corr2 = torch.sum(vx2 * vy2) / (torch.sqrt(torch.sum(vx2 ** 2)) * torch.sqrt(torch.sum(vy2 ** 2)))

            x3 = pre_th[:, 2]
            y3 = paramters_th[:, 2]
            vx3 = x3 - torch.mean(x3)
            vy3 = y3 - torch.mean(y3)
            pearson_corr3 = torch.sum(vx3 * vy3) / (torch.sqrt(torch.sum(vx3 ** 2)) * torch.sqrt(torch.sum(vy3 ** 2)))

            pearson_corr = (pearson_corr1 + pearson_corr2 + pearson_corr3) / 3

            pearson_corr_train.append(pearson_corr.item()) # 计算pcc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

########################################################## Validation ##########################################################
        with torch.no_grad():
            fast_m.eval()
            valid_load = DataLoader(dataset=valid_set, batch_size=batch, shuffle=False, num_workers=0)
            for count, data in enumerate(valid_load):
                dce_spatial, data_ref, paramters, pos = data   

                dce_spatial = dce_spatial.cuda()
                data_ref = data_ref.cuda()

                # # 3参数模型
                # # 归一化，保证对损失函数的贡献一致
                paramters[:, 0] = paramters[:, 0] / 129.86
                paramters[:, 1] = paramters[:, 1] / 655.35
                paramters[:, 2] = paramters[:, 2] / 56.0

                paramters = paramters.cuda()

                pre = fast_m(dce_spatial, data_ref)


                # ## 单参数模型 验证
                # paramters= paramters[:, 0]
                # # paramters_th = paramters[paramters<=0.05]
                # # pre_th = pre[paramters<=0.05]
                # paramters_th = paramters
                # pre_th = pre
                # pre_loss = loss_func(pre_th[:,0], paramters_th)
                #
                # v_loss = pre_loss
                # loss_array_val.append(v_loss.item())
                #
                # x = pre_th[:,0] # x表示每个批次预测的看开通ktrans值
                # y = paramters_th # y表示每个批次对应的真实的ktrans值
                # vx = x - torch.mean(x)
                # vy = y - torch.mean(y)
                # pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                # pearson_corr_val.append(pearson_corr.item())


                ### 3参数模型
                paramters_th = paramters
                pre_th = pre
                pre_loss = loss_func(pre_th, paramters_th)  # pre -> CNN predict results; parameters(obtained from NLLS fitting) -> ground truth

                loss = torch.mean(pre_loss)  # 3个参数，求均值
                loss_array_val.append(loss.item())  # add_0901，存储损失

                x1 = pre_th[:, 0]
                y1 = paramters_th[:, 0]
                vx1 = x1 - torch.mean(x1)
                vy1 = y1 - torch.mean(y1)
                pearson_corr1 = torch.sum(vx1 * vy1) / (torch.sqrt(torch.sum(vx1 ** 2)) * torch.sqrt(torch.sum(vy1 ** 2)))

                x2 = pre_th[:, 1]
                y2 = paramters_th[:, 1]
                vx2 = x2 - torch.mean(x2)
                vy2 = y2 - torch.mean(y2)
                pearson_corr2 = torch.sum(vx2 * vy2) / (torch.sqrt(torch.sum(vx2 ** 2)) * torch.sqrt(torch.sum(vy2 ** 2)))

                x3 = pre_th[:, 2]
                y3 = paramters_th[:, 2]
                vx3 = x3 - torch.mean(x3)
                vy3 = y3 - torch.mean(y3)
                pearson_corr3 = torch.sum(vx3 * vy3) / (torch.sqrt(torch.sum(vx3 ** 2)) * torch.sqrt(torch.sum(vy3 ** 2)))

                pearson_corr = (pearson_corr1 + pearson_corr2 + pearson_corr3) / 3

                pearson_corr_val.append(pearson_corr.item())  # 计算pcc

########################################################## Test ##########################################################
        # 训练模型阶段不使用测试集测试
        # with torch.no_grad(): # 测试数据集有点大,样例代码中有8万多个体素
        #     fast_m.eval()
        #     test_load = DataLoader(dataset=test_set, batch_size=batch, shuffle=False, num_workers=4) # 换成验证集
        #     for count, data in enumerate(test_load):
        #         dce_spatial, data_ref, paramters, pos = data
        #
        #         dce_spatial = dce_spatial.cuda()
        #         data_ref = data_ref.cuda()
        #         paramters = paramters.cuda()
        #
        #         pre = fast_m(dce_spatial, data_ref)
        #
        #         paramters= paramters[:, 0]
        #         # paramters_th = paramters[paramters<=0.05]
        #         # pre_th = pre[paramters<=0.05]
        #         paramters_th = paramters
        #         pre_th = pre
        #         pre_loss = loss_func(pre_th[:,0], paramters_th)
        #
        #         t_loss = pre_loss
        #         loss_array_test.append(t_loss.item())
        #
        #         x = pre_th[:,0]
        #         y = paramters_th
        #         vx = x - torch.mean(x)
        #         vy = y - torch.mean(y)
        #         pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        #         pearson_corr_test.append(pearson_corr.item())

        loss_train_avg = np.mean(loss_array_train) # 训练损失
        loss_val_avg = np.mean(loss_array_val) # 验证损失
        # loss_test_avg = np.mean(loss_array_test) # 测试损失
        pear_train_avg = np.mean(pearson_corr_train)    
        pear_val_avg = np.mean(pearson_corr_val)       
        # pear_test_avg = np.mean(pearson_corr_test)

        # # 训练集、验证集、测试集
        # print('epoch: {epoch}, train loss: {train_loss:.6f}, validation loss: {val_loss:.6f}, test loss: {test_loss:.6f}, train pearson: {pear_train_avg:.6f}, val pearson: {pear_val_avg:.6f}, test pearson: {pear_test_avg:.6f}, --stop: {stop}'.format(
        #         epoch=epoch, train_loss=loss_train_avg, val_loss=loss_val_avg, test_loss=loss_test_avg,
        #         pear_train_avg=pear_train_avg, pear_val_avg=pear_val_avg, pear_test_avg=pear_test_avg, stop=ealy_stop))        #add_0824

        # 训练集、验证集
        print('epoch: {epoch}, train loss: {train_loss:.6f}, validation loss: {val_loss:.6f}, train pearson: {pear_train_avg:.6f}, val pearson: {pear_val_avg:.6f}, --stop: {stop}'.format(
                epoch=epoch, train_loss=loss_train_avg, val_loss=loss_val_avg,
                pear_train_avg=pear_train_avg, pear_val_avg=pear_val_avg, stop=ealy_stop))        #add_0824

        #################### Learning Curve ####################
        with open(LossPath + '/learning_curve.csv', 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            # 训练集、验证集、测试集
            # wr.writerow(['%d' %(epoch), '%.15f' %loss_train_avg, '%.15f' %loss_val_avg, '%.15f' %loss_test_avg, '%.15f' %pear_train_avg, '%.15f' %pear_val_avg, '%.15f' %pear_test_avg])

            # 训练集、验证集
            wr.writerow(['%d' %(epoch), '%.15f' %loss_train_avg, '%.15f' %loss_val_avg, '%.15f' %pear_train_avg, '%.15f' %pear_val_avg])
            f.close()
        #################### Learning Curve ####################

        #################### save weight and early stopping ####################
        if min_loss > loss_val_avg: # 验证集截止当前训练epoch的最小损失，当前epoch的损失小于记录的最小损失
            best_epoch = epoch         # 将当前epoch记录下来，并存储对应的模型
            torch.save(fast_m.state_dict(), os.path.join(model_path, 'Fold'+fold, 'best_{}.tar'.format(name)))
            min_loss = loss_val_avg         
            ealy_stop = stop_num # 按这种保存方式，最后一个保存的就是最优的模型
            if best_epoch > 100 or (best_epoch <= 100 and best_epoch % 10 == 1): # 若epoch超过100
                torch.save(fast_m.state_dict(), os.path.join(model_path, 'Fold'+fold, 'best_{}.tar'.format(epoch)))
        else:                              
            ealy_stop = ealy_stop - 1    # 当前epoch的平均损失比记录的最小的损失大时，表明训练效果可能不好
        if ealy_stop and epoch < max_epoch - 1:    # 如果不是训练效果不好被停止或者是到了最后一个epoch，执行continue，不会执行下面的return语句
            continue                               
        #################### save weight and early stopping ####################

        return epoch  # 这里的return语句放在for循环内，会结束for循环并返回值，相当于只训练一次（理解错误：上述continue语句成立时不会执行该语句）

fold_df = pd.read_excel(fold_df, dtype={'subject':str, 'fold':int})
if not os.path.exists(model_path):
    os.makedirs(os.path.join(model_path)) # 创建存储模型的路径
for f in range(1, len(set(fold_df['fold']))+1): # 对5折交叉验证的模型分别存储
    if not os.path.exists(os.path.join(model_path, 'Fold' + str(f))):
        os.makedirs(os.path.join(model_path, 'Fold' + str(f)))


if ckpt_path == 0:
    print('Train the model from scratch!')

for f in range(curr_fold, curr_fold + 1):
    fast_m = CNN_ST().cuda()
    perfusion_m = perfusion().cuda() # 物理模型

    if ckpt_path != 0: # 如果要使用预训练好的模型
    #ckpt_files = glob.glob(os.path.join('exp', ckpt_path, '*.tar'))
        # ckpt_files = os.path.join('exp_3_physic_v6', ckpt_path) #
        ckpt_files = os.path.join('exp_3_physic_v6', ckpt_path)
        if os.path.isfile(ckpt_files):
            fast_m.load_state_dict(torch.load(ckpt_files))
            print('Sucessfully loaded check point from: ', ckpt_files)
        else:
            print('Wrong check point path')
            exit()

    if Optimizer_type == 'Adam':
        optimizer = Adam(fast_m.parameters(), lr=lr)
    elif Optimizer_type == 'SGD':
        optimizer = SGD(fast_m.parameters(), lr=lr)

    # train_set_pat, valid_set_pat, test_set_pat = patients_dataset(fold_df, f) # patients_dataset自定义的dataset
    # 训练和验证时不读入测试集
    train_set_pat, valid_set_pat = patients_dataset(fold_df, f)  # patients_dataset自定义的dataset

    tune_time = str(datetime.datetime.now())
    test_set_pat = [] # 训练和验证测试集为空，这里为防止程序报错
    t_epoch = train(train_set_pat, valid_set_pat, test_set_pat, fast_m, perfusion_m, optimizer, str(f), Augmentation, name='patient')

    print('Fold', f, ', Train start at', tune_time, f'total {t_epoch} epoch')
    print('Fold', f, ', End at', str(datetime.datetime.now()), f'result path:{model_path}')