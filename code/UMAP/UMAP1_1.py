import os
import random
import numpy as np
import pandas as pd
import umap
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
import torch.nn as nn
import seaborn as sns
from UMAP1 import MultiScaleConvNet
from sklearn.metrics import confusion_matrix, roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    seq = []
    labels = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split(' ')
            text = text.replace("'", "")
            label = label.replace("'", "")
            label = int(label)
            D.append((text, label))
            seq.append(text)
            labels.append(label)
    return D, seq, labels

def load_and_encoding(filename, encoding_size):
    with open(filename, encoding='utf-8') as f_in:
        lines = f_in.readlines()
    data = []
    labels = []
    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        data_nums = np.array(nums[1:]).reshape(33, encoding_size)
        data.append(data_nums)
        label_nums = nums[0]
        labels.append(label_nums)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


def onehot_encoding(seqs):
    # 确定蛋白质序列的长度，保存到一个变量中
    seq_length = len(seqs[0])
    # 初始化一个二维数组，行数为蛋白质序列的长度，列数为编码维度，每个元素都为0
    one_hot = np.zeros((seq_length, 20))
    # 遍历每个氨基酸序列，根据其在氨基酸表中的位置，在对应的列上将值设为1
    aa_table = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
                'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
    encoded_sequences = []
    for seq in seqs:
        for i, aa in enumerate(seq):
            one_hot[i, aa_table[aa]] = 1
        encoded_sequences.append(one_hot)
        one_hot = np.zeros((seq_length, 20))
    # 将编码后的二维数组保存到一个列表中，依次存储所有蛋白质序列的编码
    encoded_sequences = np.array(encoded_sequences)
    return encoded_sequences

def seed_torch(seed=999):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def umaps(data):
    myumap = umap.UMAP(n_components=2)
    output_umap = myumap.fit_transform(data)
    umap_data = np.vstack((output_umap.T, y_train)).T
    df_umap = pd.DataFrame(umap_data, columns=['Dim1', 'Dim2', 'label'])
    df_umap['label'] = df_umap['label'].replace({0: 'no', 1: 'yes'})
    df_umap.head()
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df_umap, hue='label', x='Dim1', y='Dim2')
    plt.title('Umap Visualization of train_out_data features')
    plt.legend(loc='best')
    plt.savefig('Umap visualization of train_out_data  features.jpg')
    plt.show()


if __name__ == '__main__':

    seed_torch()
    all_data, seq, label = load_data('../../datasets/ST_1.csv')
    onehot_out = onehot_encoding(seq)

    blo_out, label = load_and_encoding("../../feature/BLOSUM62/data_ST.csv", 20)
    zscale_out, label = load_and_encoding('../../feature/ZScale/data_ST.csv', 5)
    bin51_out, label = load_and_encoding('../../feature/binary/data_ST_51.csv', 5)
    bin52_out,label = load_and_encoding('../../feature/binary/data_ST_52.csv', 5)

    onehot_out = torch.from_numpy(onehot_out)
    blo_out = torch.from_numpy(blo_out)
    zscale_out = torch.from_numpy(zscale_out)
    bin51_out = torch.from_numpy(bin51_out)
    bin52_out = torch.from_numpy(bin52_out)
    label = torch.from_numpy(label)


    data = torch.cat((onehot_out, blo_out, zscale_out, bin51_out,bin52_out), dim=2)

    data = np.array(data)
    label = np.array(label)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=100)#8619,2155
    x_test = torch.from_numpy(x_test).float().to(device)
    x_train = torch.from_numpy(x_train).float().to(device)

    model = torch.load('580.8505859375.pth')
    inputs,out1,output = model(x_train)
    out = out1.detach().cpu().numpy().reshape(out1.shape[0], -1)
    umaps(out)

