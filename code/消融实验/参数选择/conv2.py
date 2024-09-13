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
        self.conv1 = nn.Conv1d(in_channels=56, out_channels=64, kernel_size=1, padding=1)

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
        out = torch.concat([out1, att_output], dim=1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
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
    one_hot = np.zeros((seq_length, 21))
    # 遍历每个氨基酸序列，根据其在氨基酸表中的位置，在对应的列上将值设为1
    aa_table = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,'X':0}
    encoded_sequences = []
    for seq in seqs:
        for i, aa in enumerate(seq):
            one_hot[i, aa_table[aa]-1] = 1
        encoded_sequences.append(one_hot)
        one_hot = np.zeros((seq_length, 21))
    # 将编码后的二维数组保存到一个列表中，依次存储所有蛋白质序列的编码
    encoded_sequences = np.array(encoded_sequences)
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
    all_data, seq, label = load_data('../../../datasets/ST_1.csv')
    onehot_out = onehot_encoding(seq)

    blo_out, label = load_and_encoding("../../../feature/BLOSUM62/data_ST.csv", 20)
    zscale_out, label = load_and_encoding('../../../feature/ZScale/data_ST.csv', 5)
    bin51_out, label = load_and_encoding('../../../feature/binary/data_ST_51.csv', 5)
    bin52_out,label = load_and_encoding('../../../feature/binary/data_ST_52.csv', 5)

    onehot_out = torch.from_numpy(onehot_out)
    blo_out = torch.from_numpy(blo_out)
    zscale_out = torch.from_numpy(zscale_out)
    bin51_out = torch.from_numpy(bin51_out)
    bin52_out = torch.from_numpy(bin52_out)
    label = torch.from_numpy(label)


    data = torch.cat((blo_out,onehot_out,bin51_out,bin52_out,zscale_out), dim=2)

    data = np.array(data)
    label = np.array(label)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=100)

    x_train1, x_valid, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.2)

    # 创建训练集和测试集的数据加载器
    train_dataset = myDataset(x_train1, y_train1)
    valid_dataset = myDataset(x_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, drop_last=True)


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

        sn,sp,mcc,acc,auc = test(model,valid_loader,device)

        if 0.82 <= acc <= 0.84 and acc > best_acc:
            torch.save(model.state_dict(), '../../../best_model/' + str(epoch) + str(acc) + '.pth')
            best_acc = acc
            print('Saved model with accuracy {:.4f}%'.format(100 * best_acc))
        print('epoch {} ,sn: {:.4f}% ,sp: {:.4f}% ,mcc: {:.4f}% ,acc: {:.4f}% ,auc: {:.4f}% '
              .format(epoch,100*sn,100*sp,100*mcc,100*acc,100*auc))
