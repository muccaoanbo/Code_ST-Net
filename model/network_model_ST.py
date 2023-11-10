from torch import nn
import torch
import numpy as np
x1 = 22.7
TE = 32
class SpatialCNN(nn.Module):
    def __init__(self):
        super(SpatialCNN, self).__init__()
        self.spatial_feature = nn.Sequential(
            nn.Conv3d(  
                in_channels=1, out_channels=8, kernel_size=(3,3,1), stride=1, padding=(1,1,0)),     #7
            nn.ReLU(),
            nn.Conv3d(  
                in_channels=8, out_channels=16, kernel_size=(3,3,1), stride=1),     
            nn.ReLU(),            
            nn.Conv3d(
                in_channels=16, out_channels=32, kernel_size=(3,3,1), stride=1),    
            nn.ReLU(),
            nn.Conv3d(
                in_channels=32, out_channels=64, kernel_size=(3,3,1), stride=1),    
            nn.ReLU(),
        )

    def forward(self, dce_spatial, data_ref):
        dce_spatial = dce_spatial.type(torch.cuda.FloatTensor)  
        data_ref = data_ref.type(torch.cuda.FloatTensor)

        # 直接拼接有bug，若批量大小为1，则会导致拼接的时候维度不匹配
        # return torch.cat([np.squeeze(self.spatial_feature(dce_spatial)), data_ref], dim=1) # 空间抽取的特征和其他的特征拼接
        return torch.cat([np.squeeze(np.squeeze(self.spatial_feature(dce_spatial), axis=2), axis=2), data_ref], dim=1)  # 空间抽取的特征和其他的特征拼接

class CNNdFeature(nn.Module):
    def __init__(self):
        super(CNNdFeature, self).__init__()
        self.feature_extra = nn.Sequential(
            nn.Conv1d(
                in_channels=66, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) # 全局和局部信息抽取前的处理
        self.local = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
        )
        self.wide = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3), 
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=8, dilation=8),  
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=25, dilation=25), 
            nn.ReLU(),
        )


    def forward(self, in_data):
        in_data = in_data.type(torch.cuda.FloatTensor)  
        cnn_feature = self.feature_extra(in_data)       
        local_feature = self.local(cnn_feature)         
        wide_feature = self.wide(cnn_feature)           
        return torch.cat([local_feature, wide_feature], dim=1) # 拼接全局特征和局部特征

class Prediction(nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()
        self.merge = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
        )
        self.pre = nn.Sequential(
            nn.Dropout(0.5),        
            nn.Linear(in_features=64*50, out_features=256), # in_features的维度从原来的64*84变成64*50
            nn.LeakyReLU(),
            nn.Dropout(0.5),        
            # nn.Linear(in_features=256, out_features=1), # 输出一个Ktrans值即可
            nn.Linear(in_features=256, out_features=3), # 3参数模型有3个输出
        )

    def forward(self, feature):
        merge_feature = self.merge(feature)    
        out = self.pre(merge_feature.view(merge_feature.size(0), 64*50))  # flatten后放入全连接层 64*84——>64*50
        return out


class CNN_ST(nn.Module):
    def __init__(self):
        super(CNN_ST, self).__init__()
        self.spatial = SpatialCNN() # 空间
        self.temporal = CNNdFeature() # 时间
        self.predict = Prediction() # 预测
    def forward(self, *args):
        return self.predict(self.temporal(self.spatial(*args)))


# 引入物理损失函数
class perfusion(nn.Module): # 用来计算信号强度的值，用于无监督的损失函数设计
    def __init__(self,):
        super(perfusion, self).__init__()
        self.ro = 1.04 # 某些参数的经验值
        self.hlv = 0.45
        self.hsv = 0.25
        self.kav = 1.4 # 依据数据集中的经验值
        self.step_size = 1  # 时间步长

    def forward(self, param, T10, cp, raw_s): # 需要传入的参数：参数值、S0（由T10决定）、动脉输入函数、静脉输出函数
        cbv = param[:, 0:1] * 5 # 得到CBV
        # 根据动脉输入和静脉输出函数，计算kav常数
        # ct = torch.zeros_like(cp) # 有一个批量维，cp表示的是动脉中信号强度的变化
        # ct = cbv / 100 * 0.73 / 1.04 * cp # 常数是基于经验值获得

        # ct = cbv / 100 * 1.04 / 0.73 * cp
        # print(ct[0, :])
        # R2 = ct / x1
        # s0 = T10
        # st = s0 * torch.exp(-TE * R2)
        # aif_t = x1 * (-1 / TE) * torch.log(cp / T10)  # 去除x1，进一步简化模型
        aif_t = (-1 / TE) * torch.log(cp / T10)
        aif_t_integral = torch.zeros(cp.size(0),1)
        for t in range(0, cp.size(-1)):
            integral = 0.0
            for i in range(1, cp.size(-1)):
                integral += (aif_t[t, i-1] + aif_t[t, i]) * self.step_size / 2.0
            aif_t_integral[t,:] = integral
        aif_t_integral = aif_t_integral.to(cbv.device)
        ct = cbv / 100 * self.ro / self.kav * (1 - self.hlv) / (1- self.hsv) * aif_t_integral
        # print(ct[0, :])
        return ct # 返回通过CBV计算得到的浓度时间曲线关于时间的积分

        # ktrans, vp, ve = param[:, 0:1]*0.2, param[:, 1:2]*0.1, param[:, 2:3]*0.6
        # ktrans = ktrans.clamp(0.00001, 0.2)
        # vp = vp.clamp(0.0005, 0.1)
        # ve = ve.clamp(0.04, 0.6)
        # ce = torch.zeros_like(cp)
        # cp_length = cp.size(-1)
        # R10 = 1 / T10  # R1 = 1 / T1
        # for t in range(cp_length):
        #     ce[:, t] = torch.sum(cp[:, :t + 1] * torch.exp(ktrans / ve * (self.range[:t + 1] - t) * deltt), dim=1) * deltt
        # ce = ce * ktrans # 对应论文中的Ce(t)函数
        # ct = vp * cp + ce # cp对应论文中的Cp(t)
        # R1 = R10 + r1 * ct # R1假设和组织中造影剂的浓度成一定的线性关系，r1是常数
        # s = (1 - torch.exp(-TR * R1)) * self.kt_sin / (1 - torch.exp(-TR * R1) * self.kt_cos) # 参数值已经的情况下，计算出信号强度
        # return s*20
    # def cuda(self):
    #     self.range = self.range.cuda()
    #     self.kt_sin = self.kt_sin.cuda()
    #     self.kt_cos = self.kt_cos.cuda()
    #     return self