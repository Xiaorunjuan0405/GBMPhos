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

        self.gru = nn.GRU(input_size=34, hidden_size=16, bidirectional=True, batch_first=True)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

        # 全连接层，将不同尺度下的特征拼接在一起并进行分类
        self.fc = nn.Linear(in_features=3104, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)


    def forward(self, x,batch):
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

        out2_1 = self.drop(out2_1)
        # print(out2_1.shape)
        out2_2 = self.drop(out2_2)
        # print(out2_1.shape)
        out2_3 = self.drop(out2_3)
        # print(out2_1.shape)

        gate_out1 = torch.sigmoid(out2_1)
        gate_out2 = torch.sigmoid(out2_2)
        gate_out3 = torch.sigmoid(out2_3)

        gate_out1 =gate_out1*(1-gate_out2)*gate_out3# 32,32,33
        att_output = gate_out1.reshape(batch, -1)

        gru_out1, (_, _) = self.gru(out1)  #
        gru_out1 = gru_out1.reshape(batch, -1)

        # 将不同尺度下的特征进行分类
        out = self.fc(torch.cat([gru_out1, att_output], dim=1))
        out = self.fc1(out)
        # out = self.drop(out)
        out = self.fc2(out)
        # out = self.drop(out)
        out = self.sig(out)

        return out1,gate_out1,gru_out1,out

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
            out1,gate_out1,gru_out1,output = model(data,128)
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

    model = torch.load('E:\\myarticle\\best_model\\580.8505859375.pth')
    conv1_out,gate_out1,gru_out1,out = model(x_test,2155)
    out = out.detach().cpu().numpy().reshape(out.shape[0],-1)
    myumap = umap.UMAP(n_components=2)
    out_umap = myumap.fit_transform(out)
    out_umap_data = np.vstack((out_umap.T,y_test)).T
    df_umap = pd.DataFrame(out_umap_data, columns=['Dim1', 'Dim2', 'label'])
    df_umap.head()
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df_umap, hue='label', x='Dim1', y='Dim2')
    plt.title('Umap Visualization of  out features')
    plt.legend(loc='best')
    plt.savefig('Umap visualization of out features.jpg')
    plt.show()



    # model = MultiScaleConvNet().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-05)
    # criterion = nn.BCELoss()


    # # 训练和测试模型
    # for epoch in range(40):
    #     best_acc=0
    #     model.train()
    #     avg_weights=[]
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         data = data.float().to(device)
    #         target = target.float().to(device)
    #         # 前向传播
    #         out1,gate_out1,gru_out1,output = model(data,128)
    #         # 计算损失
    #         # L1正则化
    #         loss = loss_fn(output, target, model, 0.001)
    #         # 反向传播和参数更新
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
        # if epoch==37:
        #     conv_out = conv_out.detach().cpu().numpy().reshape(conv_out.shape[0],-1)
        #     target = target.detach().cpu().numpy()
        #     myumap = umap.UMAP(n_components=2)
        #     out_umap = myumap.fit_transform(conv_out)
        #     fc_out_umap_data = np.vstack((out_umap.T,target)).T
        #     df_umap = pd.DataFrame(fc_out_umap_data, columns=['Dim1', 'Dim2', 'label'])
        #     df_umap.head()
        #     plt.figure(figsize=(6, 6))
        #     sns.scatterplot(data=df_umap, hue='label', x='Dim1', y='Dim2')
        #     plt.title('Umap Visualization of  conv1 features')
        #     plt.legend(loc='best')
        #     plt.savefig('Umap visualization of conv1 features.jpg')
        #     plt.show()



        # sn,sp,mcc,acc,auc = test(model,test_loader,device)
        #
        # if acc > 0.857 and acc > best_acc:
        #     torch.save(model, '../best_model/' + str(epoch) + str(acc) + '.pth')
        #     best_acc = acc
        #     print('Saved model with accuracy {:.4f}%'.format(100 * best_acc))
        # print('epoch {} ,sn: {:.4f}% ,sp: {:.4f}% ,mcc: {:.4f}% ,acc: {:.4f}% ,auc: {:.4f}% '
        #       .format(epoch,100*sn,100*sp,100*mcc,100*acc,100*auc))
