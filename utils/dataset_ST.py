from torch.utils.data import Dataset, ConcatDataset, random_split
import torch
import os
import numpy as np
import random
from utils.config import get_config
import nibabel as nib
from torchvision import transforms
import torchio as tio



config = get_config()

data_root = config.train.dataset # 训练数据
train_val_ratio = config.train.train_val_ratio
test_path = config.test.dataset # 测试数据存储路径

torch.manual_seed(202109)
torch.cuda.manual_seed_all(202109)
np.random.seed(202109)


class Patient(Dataset):
    def __init__(self, root):
        self.root = root
        self.data = []
        self.cp = self.read_cp() # 读取造影剂在时间序列上的变化
        self.read_data()

    def __getitem__(self, item):
        data = self.data[item] # 逐个处理.npy数据中存储的数据
        pos = data.get('position')  # 记录的是voxel的坐标
        T10 = np.array([data.get('T10'),])   # T10表示的是信号强度值
        raw_data = data.get('dce_data')
        cp = self.cp # 肌肉中造影剂时间序列上的平均浓度，84 * 1
        #Ktrans_all = np.load('/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/paper/new/Ktrans_High_recreate_Th_npy/' + self.root[-5:] + '.npy', allow_pickle=True)
        #Ktrans_p = Ktrans_all[pos[0], pos[1], pos[2]]
        #param = np.array([Ktrans_p, Ktrans_p, Ktrans_p])
        param = data.get('param') # ktrans参数，在训练和测试集存储的都是单个体素对应的ktrans值

##################### add 0429 #########################
        # T10_pcr99 = np.load('/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/DL_input/T10_prc99/' + self.root[-5:] + '.npy', allow_pickle=True)
        # 构建物理损失函数模型时，这里不对原始数据做归一化
        T10_pcr99 = np.load('/data0/BME_caoanbo/project_coding/K-trans_STNet/data/DL_input/T10_prc99/' + self.root[-15:] + '.npy', allow_pickle=True)
        # T10 = T10 / T10_pcr99 # 相当于归一化操作
        # dce_spatial = raw_data / T10_pcr99
        # cp = cp / T10_pcr99
        dce_spatial = raw_data

        extra_data = np.ones_like(cp) # 创建一个与cp相同shape的数组，元素值初始化为1
        extra_data = extra_data*T10

        cp = cp[np.newaxis, ...] # 新增一个维度

        data_ref = np.concatenate((cp, np.reshape(extra_data,(1,cp.shape[1]))), axis=0)
        dce_spatial = dce_spatial[np.newaxis, ...]  # 1, 7, 7, 84，裁剪的patches
        return dce_spatial.astype(np.float), data_ref.astype(np.float), param.astype(np.float32), pos           

    def __len__(self):
        return len(self.data)

    def read_cp(self):
        info_file = os.path.join(self.root, 'CA.npy')
        cp = np.load(info_file, allow_pickle=True)
        return cp

    def read_data(self): # 读取数据，包含训练集等
        eTofts_files = os.listdir(self.root)
        eTofts_files.remove('CA.npy') # 不读CA.npy,CA是肌肉中造影剂的浓度值
        eTofts_files.remove('VCA.npy')  # 不读CA.npy,CA是肌肉中造影剂的浓度值
        eTofts_files.sort(key=lambda x: int((os.path.splitext(x)[0]).split('_')[1]))
        for file in eTofts_files:
            data = np.load(os.path.join(self.root, file), allow_pickle=True)
            self.data.extend(data) # 数据读入到类的data属性中

def patients_dataset(fold_df, f):
    testing_subject = fold_df.loc[fold_df['fold'] == f, ['subject']]['subject'].tolist()
    training_subject = fold_df['subject'].tolist()
    print(training_subject)


    patients = []
    test_set = []
    print('Testing path: ', test_path)
    print('Training path: ', data_root)
    # for pats in training_subject:
    #     if pats in testing_subject: # 测试集 实验中指定为3
    #         print('Loading testing subjects: ', pats)
    #         test_set.append(Patient(os.path.join(test_path, pats)))    # 根据ID的不同读入不同个体的数据
    #     else: # 训练和验证集
    #         print('Loading training + validation subjects: ', pats)
    #         patients.append(Patient(os.path.join(data_root, pats)))

    # 此处为了跑样例程序,不做交叉验证,训练集,验证集和测试集的数据来自同一个个体 ID002
    for pats in training_subject:
        if pats in testing_subject: # 测试集 实验中指定为3
            # 训练和验证时不读入测试集
            # print('Loading testing subjects: ', pats)
            # test_set.append(Patient(os.path.join(test_path, pats)))    # 根据ID的不同读入不同个体的数据
        # else: # 训练和验证集
            print('Loading training + validation subjects: ', pats)
            # print(os.path.join(data_root, pats))
            patients.append(Patient(os.path.join(data_root, pats)))

    # test = ConcatDataset(test_set)
    sets = ConcatDataset(patients)                                                         
    train_len = int(sets.__len__()*train_val_ratio)                                                 
    train, valid = random_split(sets, [train_len, sets.__len__()-train_len])    # 划分训练集和测试集
    # return train, valid, test
    return train, valid




def dataAug(data): # 数据增强
    dce_spatial = data
    dce_Aug = np.zeros(dce_spatial.shape)
    # affine_transform = tio.RandomAffine()
    training_transform = tio.Compose({
                                        tio.OneOf({
                                                    tio.RandomBlur(),                 
                                                    tio.RandomNoise()
                                                    }, p=0.5),
                                        # tio.RandomMotion(p=0.5),
                                        # tio.RandomAffine(p=0.5),

        })

    for b in range(dce_spatial.shape[0]):
        dce_spatial_4D = dce_spatial[b]   # (1, x, y, t)
        dce_spatial_4D_Aug = training_transform(dce_spatial_4D)
        dce_Aug[b, :, :, :, :] = dce_spatial_4D_Aug
    # dce_spatial_4D = dce_spatial[np.newaxis, ...]
    # dce_spatial_4D_Aug = training_transform(dce_spatial_4D)
    return torch.tensor(dce_Aug)
