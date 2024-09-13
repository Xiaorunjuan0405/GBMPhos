import csv
import math
import os
import random
import numpy as np
from matplotlib import cm, colors
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import torch
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import warnings
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
import torch.nn as nn

from sklearn.metrics import confusion_matrix, roc_auc_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#cyj
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads

        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x):
        batch_size, seq_len, input_size= x.shape

        # Perform linear transformation and split into num_heads
        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute the attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())

        # Normalize the attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)


        # Compute the weighted average of the values
        context = torch.matmul(attention_probs, values).transpose(1, 2).contiguous().view(batch_size, seq_len, input_size)

        # Apply a linear transformation and return the output
        output = self.fc(context)

        return output,attention_probs #32,33,32    32,2,33,33



class MultiScaleConvNet(nn.Module):
    def __init__(self):
        super(MultiScaleConvNet, self).__init__()

        # 第一层卷积，对原始序列进行卷积
        self.conv1 = nn.Conv1d(in_channels=55, out_channels=64, kernel_size=1, padding=1)

        # 第二层卷积，对不同尺度下的序列进行卷积
        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        # 池化层，将不同尺度下的特征进行池化
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)

        self.gru = nn.GRU(input_size=34, hidden_size=16, bidirectional=True, batch_first=True)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.atten = MultiHeadAttention(32,2)


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

        # 全连接层，将不同尺度下的特征拼接在一起并进行分类
        self.fc = nn.Linear(in_features=3104, out_features=128)
        self.fc1 = nn.Linear(in_features=128, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=1)


    def forward(self, x,batch_size):
        inputs = x.transpose(dim0=1, dim1=2)  # torch.Size([1 ,5, 33])
        # 第一层卷积
        out1 = self.conv1(inputs)
        out1 = self.pool(out1)
        out1 = self.bn1(out1)

        # 第二层卷积
        out2_1 = self.conv2_1(out1)
        # print(out2_1.shape)
        out2_2 = self.conv2_2(out1)
        # print(out2_2.shape)
        out2_3 = self.conv2_3(out1)
        # print(out2_3.shape)

        # 对不同尺度下的特征进行池化
        out2_1 = self.pool(out2_1)
        # print(out2_1.shape)
        out2_2 = self.pool(out2_2)
        # print(out2_1.shape)
        out2_3 = self.pool(out2_3)
        # print(out2_1.shape)

        out2_1 = self.bn2(out2_1)
        # print(out2_1.shape)
        out2_2 = self.bn2(out2_2)
        # print(out2_1.shape)
        out2_3 = self.bn2(out2_3)
        # print(out2_1.shape)

        gate_out1 = torch.sigmoid(out2_1)
        gate_out2 = torch.sigmoid(out2_2)
        gate_out3 = torch.sigmoid(out2_3)

        gate_out1 = gate_out1*(1-gate_out2)*gate_out3  # 32,32,33
        gate_out = gate_out1.transpose(dim0=1, dim1=2)  # torch.Size([32 ,33, 32])
        # gate_out = gate_out.transpose(dim0=0, dim1=1)#33,32,32
        att_output,att_weight = self.atten(gate_out) #33,32,32    33,1,32,32
        # 将输出张量转换回 (batch_size, seq_len, input_dim) 的形式
        # att_output = att_output.permute(1, 0, 2)
        # print(att_output.shape)
        att_output = att_output.reshape(batch_size, -1)

        out1, (_, _) = self.gru(out1)  #
        out1 = out1.reshape(batch_size, -1)

        # 将不同尺度下的特征进行分类
        out = self.fc(torch.cat([out1, att_output], dim=1))
        out = self.fc1(out)
        # out = self.drop(out)
        out = self.fc2(out)
        # out = self.drop(out)
        out = self.sig(out)

        return out,att_weight

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


class myDataset(Dataset):
    def __init__(self, seq, label):
        self.protein_seq = seq
        self.label_list = label

    def __getitem__(self, index):
        seq = self.protein_seq[index]
        label = self.label_list[index]
        return seq, label

    def __len__(self):
        return len(self.protein_seq)


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



def loss_fn(output, target, model, lambda_):  #https://cloud.tencent.com/developer/ask/sof/149180  https://zhuanlan.zhihu.com/p/259159952
    output = output.squeeze()

    # 计算模型的原始损失函数
    loss = criterion(output, target)
    # 计算L1正则化的惩罚项
    l1_loss = torch.tensor(0.)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_loss = l1_loss + torch.norm(param, p=1)
    loss = loss + lambda_ * l1_loss

    return loss



def test(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.float().to(device)
            target = target.float().to(device)
            output,att_weight = model(data,32)
            output = output.squeeze()

            y_true += target.cpu()
            y_pred += (output > 0.5).cpu()
            y_score += output.cpu()


    y_pred = [float(x) for x in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)

    y_pred_np = torch.tensor(y_pred).cpu().numpy()
    y_true_np = torch.tensor(y_true).cpu().numpy()

    acc = np.count_nonzero(y_pred_np == y_true_np) / len(y_true)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    auc = roc_auc_score(y_true, y_score)

    return sn, sp, mcc, acc, auc


def plot(weight):
    att_weight = weight.detach().cpu().numpy()

    # att_weight = np.mean(att_weight, axis=0)

    # 创建一个新的图表
    fig, ax = plt.subplots()
    for i in att_weight:
        x = range(33)
        y = i

        ax.set_ylim(0.03020, 0.03040)
        ax.set_yticks([0.03020, 0.03021, 0.03022,0.03023,0.03024,0.03025,0.03026,
                       0.03027, 0.03028, 0.03029, 0.03030, 0.03031, 0.03032, 0.03033,
                       0.03034, 0.03035,0.03036,0.03037,0.03038,0.03039,0.03040])
        # 画条形图
        ax.bar(x, y)

        # 添加标题和标签
        plt.title('Attention')
        plt.xlabel('length')
        plt.ylabel('Value')

        # 显示图形
        plt.show()


def seed_torch(seed=999):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    seed_torch()
    all_data, seq, label = load_data('../../datasets/ST_1.csv')
    onehot_out = onehot_encoding(seq)

    blo_out, label = load_and_encoding("../../feature/BLOSUM62/data_ST.csv",20)
    zscale_out, label = load_and_encoding('../../feature/ZScale/data_ST.csv',5)
    bin51_out, label = load_and_encoding('../../feature/binary/data_ST_51.csv',5)
    bin52_out,label = load_and_encoding('../../feature/binary/data_ST_52.csv',5)

    onehot_out = torch.from_numpy(onehot_out)
    blo_out = torch.from_numpy(blo_out)
    zscale_out = torch.from_numpy(zscale_out)
    bin51_out = torch.from_numpy(bin51_out)
    bin52_out = torch.from_numpy(bin52_out)
    label = torch.from_numpy(label)


    data = torch.cat((onehot_out, blo_out, zscale_out, bin51_out,bin52_out), dim=2)

    data = np.array(data)
    label = np.array(label)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=100)

    positive_indices = y_test == 1
    negative_indices = y_test == 0

    x_positive = x_test[positive_indices]
    y_positive = y_test[positive_indices]

    x_negative = x_test[negative_indices]
    y_negative = y_test[negative_indices]

    # 创建训练集和测试集的数据加载器
    x_positive_dataset = myDataset(x_positive, y_positive)
    x_negative_dataset = myDataset(x_negative, y_negative)

    x_positive_dataset_loader = DataLoader(x_positive_dataset, batch_size=32, shuffle=True, drop_last=True)
    x_negative_dataset_loader = DataLoader(x_negative_dataset, batch_size=32, shuffle=False, drop_last=True)


    model =torch.load('E:\myarticle\\best_model\\340.8568097014925373.pth').to(device)

    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    attn_weights_list = []
    with torch.no_grad():
        for data, target in x_negative_dataset_loader:
            data = data.float().to(device)
            target = target.float().to(device)
            output, attn_weights = model(data, 32)
            output = output.squeeze()

            y_true += target.cpu()
            y_pred += (output > 0.5).cpu()
            y_score += output.cpu()
            attn_weights_list.append(attn_weights)

    attn_weights_tensor = torch.cat(attn_weights_list, dim=0)
    num_heads = attn_weights_tensor.shape[1]

    plt.subplots_adjust(hspace=0.6)  # 调整纵向子图之间的间距
    plt.figure(figsize=(24, 24))
    # 遍历每个注意力头
    for i in range(num_heads):
        attn_matrix = attn_weights_tensor[:, i].detach().cpu().numpy().mean(axis=0)
        plt.subplot(num_heads + 1, 1, i + 1)
        plt.imshow(attn_matrix, cmap='Blues', vmin=attn_matrix.min(), vmax=attn_matrix.max())
        plt.title('Attention weights - head {}'.format(i + 1))
        plt.colorbar()

    # 计算平均注意力权重并画出对应热图
    mean_attn_weights = attn_weights_tensor.mean(dim=1).detach().cpu().numpy().mean(axis=0)
    plt.subplot(num_heads + 1, 1, num_heads + 1)
    plt.imshow(mean_attn_weights, cmap='Blues', vmin=mean_attn_weights.min(), vmax=mean_attn_weights.max())
    plt.title('Mean attention weights')
    plt.colorbar()
    plt.show()

    # y_pred = [float(x) for x in y_pred]
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # sn = tp / (tp + fn)
    # sp = tn / (tn + fp)
    #
    # y_pred_np = torch.tensor(y_pred).cpu().numpy()
    # y_true_np = torch.tensor(y_true).cpu().numpy()
    #
    # acc = np.count_nonzero(y_pred_np == y_true_np) / len(y_true)
    # mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    # auc = roc_auc_score(y_true, y_score)




