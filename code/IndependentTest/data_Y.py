import os
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        self.gru = nn.GRU(input_size=64, hidden_size=32, bidirectional=True, batch_first=True)

        self.sig = nn.Sigmoid()
        self.flattten = nn.Flatten()
        self.drop = nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)


        # 全连接层，将不同尺度下的特征拼接在一起并进行分类
        self.fc = nn.Linear(in_features=3232, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)


    def forward(self, x):
        # print(x.shape)
        out1 = self.conv1(x)
        out1 = self.pool(out1)
        out1 = self.bn1(out1)  #128,64,34

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

        out2_1 = self.drop(out2_1)
        # print(out2_1.shape)
        out2_2 = self.drop(out2_2)
        # print(out2_1.shape)
        out2_3 = self.drop(out2_3)
        # print(out2_1.shape)

        gate_out1 = self.sig(out2_1)
        gate_out2 = self.sig(out2_2)
        gate_out3 = self.sig(out2_3)

        gate_out1 =gate_out1*(1-gate_out2)*gate_out3# 32,32,33
        att_output = self.flattten(gate_out1)
        # att_output = self.ln1(att_output)

        out1 = out1.transpose(dim0=1, dim1=2)
        out1, (_, _) = self.gru(out1)  #
        out1 = self.flattten(out1)
        # out1 = self.ln2(out1)

        # 将不同尺度下的特征进行分类
        out = torch.cat([out1, att_output], dim=1)

        out = self.fc(out)
        out = self.fc1(out)
        # out = self.drop(out)
        out = self.fc2(out)
        # out = self.drop(out)
        out = self.sig(out)

        return out

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
    aa_table = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}
    encoded_sequences = []
    for seq in seqs:
        for i, aa in enumerate(seq):
            one_hot[i, aa_table[aa]-1] = 1
        encoded_sequences.append(one_hot)
        one_hot = np.zeros((seq_length, 20))
    # 将编码后的二维数组保存到一个列表中，依次存储所有蛋白质序列的编码
    encoded_sequences = np.array(encoded_sequences)
    return encoded_sequences

def zscale_encode_sequences(file_path):
    zscale = {
        'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
        'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
        'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
        'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
        'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
        'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
        'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
        'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
        'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
        'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
        'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
        'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
        'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
        'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
        'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
        'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
        'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
        'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
        'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
        'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
        '-': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
    }

    encoded_sequences = []

    with open(file_path, 'r') as file:
        current_sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    encoded_sequence = np.array([zscale[aa] for aa in current_sequence])
                    encoded_sequences.append(encoded_sequence)
                current_name = line
                current_sequence = ''
            else:
                current_sequence += line

        # 处理最后一个序列
        if current_sequence:
            encoded_sequence = np.array([zscale[aa] for aa in current_sequence])
            encoded_sequences.append(encoded_sequence)
    return encoded_sequences

def binary51_encode_sequences(file_path):
    zscale = {
        'G': [1, 0, 0, 0, 0],  # A
        'A': [1, 0, 0, 0, 0],  # C
        'V': [1, 0, 0, 0, 0],  # D
        'L': [1, 0, 0, 0, 0],  # E
        'M': [1, 0, 0, 0, 0],  # F
        'I': [1, 0, 0, 0, 0],  # G
        'F': [0, 1, 0, 0, 0],  # H
        'Y': [0, 1, 0, 0, 0],  # I
        'W': [0, 1, 0, 0, 0],  # K
        'K': [0, 0, 1, 0, 0],  # L
        'R': [0, 0, 1, 0, 0],  # M
        'H': [0, 0, 1, 0, 0],  # N
        'D': [0, 0, 0, 1, 0],  # P
        'E': [0, 0, 0, 1, 0],  # Q
        'S': [0, 0, 0, 0, 1],  # R
        'T': [0, 0, 0, 0, 1],  # S
        'C': [0, 0, 0, 0, 1],  # T
        'P': [0, 0, 0, 0, 1],  # V
        'N': [0, 0, 0, 0, 1],  # W
        'Q': [0, 0, 0, 0, 1],  # Y
        '-': [0, 0, 0, 0, 0],  # -
    }

    encoded_sequences = []

    with open(file_path, 'r') as file:
        current_sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    encoded_sequence = np.array([zscale[aa] for aa in current_sequence])
                    encoded_sequences.append(encoded_sequence)
                current_name = line
                current_sequence = ''
            else:
                current_sequence += line

        # 处理最后一个序列
        if current_sequence:
            encoded_sequence = np.array([zscale[aa] for aa in current_sequence])
            encoded_sequences.append(encoded_sequence)
    return encoded_sequences

def binary52_encode_sequences(file_path):
    zscale = {
        'A': [0, 0, 0, 1, 1],  # A
        'C': [0, 0, 1, 0, 1],  # C
        'D': [0, 0, 1, 1, 0],  # D
        'E': [0, 0, 1, 1, 1],  # E
        'F': [0, 1, 0, 0, 1],  # F
        'G': [0, 1, 0, 1, 0],  # G
        'H': [0, 1, 0, 1, 1],  # H
        'I': [0, 1, 1, 0, 0],  # I
        'K': [0, 1, 1, 0, 1],  # K
        'L': [0, 1, 1, 1, 0],  # L
        'M': [1, 0, 0, 0, 1],  # M
        'N': [1, 0, 0, 1, 0],  # N
        'P': [1, 0, 0, 1, 1],  # P
        'Q': [1, 0, 1, 0, 0],  # Q
        'R': [1, 0, 1, 0, 1],  # R
        'S': [1, 0, 1, 1, 0],  # S
        'T': [1, 1, 0, 0, 0],  # T
        'V': [1, 1, 0, 0, 1],  # V
        'W': [1, 1, 0, 1, 0],  # W
        'Y': [1, 1, 1, 0, 0]  # Y
    }

    encoded_sequences = []

    with open(file_path, 'r') as file:
        current_sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    encoded_sequence = np.array([zscale[aa] for aa in current_sequence])
                    encoded_sequences.append(encoded_sequence)
                current_name = line
                current_sequence = ''
            else:
                current_sequence += line

        # 处理最后一个序列
        if current_sequence:
            encoded_sequence = np.array([zscale[aa] for aa in current_sequence])
            encoded_sequences.append(encoded_sequence)
    return encoded_sequences

def blosum62_encode_sequences(file_path):
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    encoded_sequences = []

    with open(file_path, 'r') as file:
        current_sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    encoded_sequence = np.array([blosum62[aa] for aa in current_sequence])
                    encoded_sequences.append(encoded_sequence)
                current_name = line
                current_sequence = ''
            else:
                current_sequence += line

        # 处理最后一个序列
        if current_sequence:
            encoded_sequence = np.array([blosum62[aa] for aa in current_sequence])
            encoded_sequences.append(encoded_sequence)
    return encoded_sequences

def test(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.float().to(device)
            inputs = data.transpose(dim0=1, dim1=2)  # 128,55,33
            target = target.float().to(device)
            output = model(inputs)
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

def L1Loss(model,beta):  #https://blog.csdn.net/Zzz_zhongqing/article/details/107528717
    l1_loss = torch.tensor(0.0,requires_grad = True)
    for name,parma in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(parma))
    return l1_loss


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


    blo_out = blosum62_encode_sequences("../../datasets/ST.fasta")
    zscale_out = zscale_encode_sequences("../../datasets/ST.fasta")
    bin51_out= binary51_encode_sequences("../../datasets/ST.fasta")
    bin52_out= binary52_encode_sequences("../../datasets/ST.fasta")

    onehot_out = torch.from_numpy(onehot_out)
    blo_out = torch.tensor(blo_out)
    zscale_out = torch.tensor(zscale_out)
    bin51_out = torch.tensor(bin51_out)
    bin52_out = torch.tensor(bin52_out)



    data = torch.cat((onehot_out, blo_out, zscale_out, bin51_out,bin52_out), dim=2)


    data = np.array(data)
    label = np.array(label)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=66)
    # print(x_train.shape)

    # 创建训练集和测试集的数据加载器
    train_dataset = myDataset(x_train, y_train)
    test_dataset = myDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True)


    model = MultiScaleConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-05)
    criterion = nn.BCELoss()


    # 训练和测试模型
    for epoch in range(100):
        best_acc=0
        model.train()
        avg_weights=[]
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().to(device)
            inputs = data.transpose(dim0=1, dim1=2)  # 128,55,33
            target = target.float().to(device)
            # 前向传播
            output = model(inputs)
            # 计算损失
            # L1正则化
            output = output.squeeze()
            loss = criterion(output,target)
            loss = loss+ L1Loss(model,0.001)
            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sn,sp,mcc,acc,auc = test(model,test_loader,device)

        if acc > 0.846 and acc > best_acc:
            torch.save(model, '../../best_model/' + str(epoch) + str(acc) + '.pth')
            # torch.save(model.state_dict(),'../../best_model1/BiGRU_model_weights.pt')
            best_acc = acc
            print('Saved model with accuracy {:.4f}%'.format(100 * best_acc))
        print('epoch {} ,sn: {:.4f}% ,sp: {:.4f}% ,mcc: {:.4f}% ,acc: {:.4f}% ,auc: {:.4f}% '
              .format(epoch,100*sn,100*sp,100*mcc,100*acc,100*auc))
