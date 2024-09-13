import os
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import warnings
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiScaleConvNet(nn.Module):
    def __init__(self):
        super(MultiScaleConvNet, self).__init__()

        # 第一层卷积，对原始序列进行卷积
        self.conv1 = nn.Conv1d(55, 16, kernel_size=3, padding=1)

        # 第二层卷积，对不同尺度下的序列进行卷积
        self.conv2_1 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv1d(16, 8, kernel_size=3, padding=1)

        # 池化层，将不同尺度下的特征进行池化
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)

        self.gru = nn.GRU(input_size=32, hidden_size=10, bidirectional=True, batch_first=True)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.attfc1 = nn.Linear(31, 8)
        self.attfc2 = nn.Linear(8, 1)

        # self.ln = nn.LayerNorm(2048)
        self.drop = nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(8)

        # 全连接层，将不同尺度下的特征拼接在一起并进行分类
        self.fc = nn.Linear(351, 64)
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8,1)

    def attention(self,input):
        input = input.transpose(dim0=1,dim1=2)
        # print(input.shape)
        # input: tensor of shape (batch_size, sequence_length, input_dim)
        # hidden_dim: the dimension of the hidden representation of the attention mechanism

        # Apply linear transformation to the input tensor
        linear_transform = nn.Linear(34, 8, bias=False)
        u = self.attfc1(input)  # (batch_size, sequence_length, hidden_dim)

        # Apply tanh activation function to the transformed input
        tanh_activation = nn.Tanh()
        u = tanh_activation(u)  # (batch_size, sequence_length, hidden_dim)

        # Apply another linear transformation to the transformed input
        linear_transform = nn.Linear(8, 1, bias=False)
        alpha = self.attfc2(u)  # (batch_size, sequence_length, 1)

        # Apply softmax activation function to the alpha tensor
        softmax_activation = nn.Softmax(dim=1)
        alpha = softmax_activation(alpha)  # (batch_size, sequence_length, 1)

        # Apply the attention weights to the input tensor to get the output
        output = torch.sum(alpha * input, dim=1)  # (batch_size, input_dim)

        return output

    def forward(self, x):
        inputs = x.transpose(dim0=1, dim1=2)  # torch.Size([1 ,5, 33])
        inputs = torch.tensor(inputs)
        inputs = inputs.to(torch.float32)
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

        # out2_1 = self.relu(out2_1)
        # # print(out2_1.shape)
        # out2_2 = self.relu(out2_2)
        # # print(out2_1.shape)
        # out2_3 = self.relu(out2_3)
        # # print(out2_1.shape)

        out2_1 = self.bn2(out2_1)
        # print(out2_1.shape)
        out2_2 = self.bn2(out2_2)
        # print(out2_1.shape)
        out2_3 = self.bn2(out2_3)
        # print(out2_1.shape)

        gate_out1 = torch.sigmoid(out2_1)
        gate_out2 = torch.sigmoid(out2_2)
        gate_out3 = torch.sigmoid(out2_3)

        gate_out1 = gate_out1*(1-gate_out2)*gate_out3  # 16,32,33
        # gate_out2 = gate_out1*(1-gate_out2)*gate_out3
        # gate_out3 = gate_out1*gate_out2*(1-gate_out3)
        # # 将不同尺度下的特征拼接在一起
        # out2 = torch.cat([out2_1, out2_2, out2_3], dim=1)#16,32,96
        # gate_out = torch.sigmoid(out2)
        # gate_out = gate_out*out2

        # lstm_output = self.lstm(gate_out)
        gate_out = gate_out1.transpose(dim0=1, dim1=2)  # torch.Size([16 ,33, 32])
        att_output = self.attention(gate_out)
        att_output = att_output.reshape(2, -1)

        out1, (_, _) = self.gru(out1)  #
        out1 = out1.reshape(2, -1)


        # out1 = self.ln(out1)

        # 将不同尺度下的特征进行分类
        out = self.fc(torch.cat([out1, att_output], dim=1))
        out = self.fc1(out)
        # out = self.drop(out)
        out = self.fc2(out)
        # out = self.drop(out)
        out = self.sig(out)

        return out


# def load_feature(filename):


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


def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    Pre = TP / (TP + FP)

    return SN, SP, ACC, MCC, Pre


def cal_score(pred, label):
    try:
        AUC = roc_auc_score(list(label), pred)
    except:
        AUC = 0

    pred = np.around(pred)
    label = np.array(label)

    confus_matrix = confusion_matrix(label, pred, labels=None, sample_weight=None)
    SN, SP, ACC, MCC, Pre = Model_Evaluate(confus_matrix)
    print(
        "Model score --- SN:{0:.3f}       SP:{1:.3f}       ACC:{2:.3f}       MCC:{3:.3f}      Pre:{4:.3f}   AUC:{5:.3f}".format(
            SN, SP, ACC, MCC, Pre, AUC))

    return ACC


def fit(model, train_loader, optimizer, criterion):
    model.train()
    pred_list = []
    label_list = []

    for seq, label in train_loader:
        label = torch.tensor(label).float()
        pred = model(seq)
        pred = pred.squeeze()
        # print(pred)
        loss = criterion(pred, label)

        l1 = torch.tensor(0.)
        for param in model.parameters():
            l1 += torch.norm(param, 1)
        loss += 0.01 * l1  # 添加l1正则化项

        model.zero_grad()
        loss.backward()
        optimizer.step()
        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score


def validate(model, val_loader):
    model.eval()
    pred_list = []
    label_list = []

    for seq, label in val_loader:
        label = torch.tensor(label).float()

        pred = model(seq)

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score, pred_list, label_list


def load_and_encoding_BLO(filename):
    with open(filename, encoding='utf-8') as f_in:
        lines = f_in.readlines()
    data = []
    labels = []
    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        nums = np.array(nums)
        nums = nums[1:].reshape(33, 20)
        data.append(nums)

    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        nums = torch.tensor(nums[0])
        labels.append(nums)
    return data, labels

def load_and_encoding_ZCscale(filename):
    with open(filename, encoding='utf-8') as f_in:
        lines = f_in.readlines()
    data = []
    labels = []
    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        nums = np.array(nums[1:]).reshape(33, 5)
        data.append(nums)
    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        nums = torch.tensor(nums[0])
        labels.append(nums)
    return data, labels

def load_and_encoding_bin52(filename):
    with open(filename, encoding='utf-8') as f_in:
        lines = f_in.readlines()
    data = []
    labels = []
    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        nums = np.array(nums[1:]).reshape(33, 5)
        data.append(nums)
    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        nums = torch.tensor(nums[0])
        labels.append(nums)
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


def load_and_encoding_bin51(filename):
    with open(filename, encoding='utf-8') as f_in:
        lines = f_in.readlines()
    data = []
    labels = []
    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        nums = np.array(nums[1:]).reshape(33, 5)
        data.append(nums)
    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        nums = torch.tensor(nums[0])
        labels.append(nums)
    return data, labels

def load_and_encoding_bin31(filename):
    with open(filename, encoding='utf-8') as f_in:
        lines = f_in.readlines()
    data = []
    labels = []
    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        nums = np.array(nums[1:]).reshape(33, 3)
        data.append(nums)
    for line in lines:
        nums = [float(num) for num in line.strip().split(',')]
        nums = torch.tensor(nums[0])
        labels.append(nums)
    return data, labels

def loss_fn(output, target, model, lambda_):
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
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            output = output.squeeze()
            y_true += target.cpu().tolist()
            y_pred += (output > 0.5).cpu().tolist()
            y_score += output.cpu().tolist()


    y_pred = [float(x) for x in y_pred]
    # print(y_true)
    # print(y_pred)
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



    all_data, seq, label = load_data('../../datasets/Y_1.csv')
    onehot_out = onehot_encoding(seq)

    blo_out, label = load_and_encoding_BLO("../../feature/BLOSUM62/data_Y.csv")
    blo_out = np.array(blo_out)

    zscale_out, label = load_and_encoding_ZCscale('../../feature/ZScale/data_Y.csv')
    zscale_out = np.array(list(zscale_out))

    bin51_out, label = load_and_encoding_bin51('../../feature/binary/data_Y_51.csv')
    bin51_out = np.array(bin51_out)

    bin52_out,label = load_and_encoding_bin52('../../feature/binary/data_Y_52.csv')
    bin52_out = np.array(bin52_out)


    # zscale_out = zscale_out.to(torch.float32)
    onehot_out = torch.from_numpy(onehot_out)
    blo_out = torch.from_numpy(blo_out)
    zscale_out = torch.from_numpy(zscale_out)
    bin51_out = torch.from_numpy(bin51_out)
    bin52_out = torch.from_numpy(bin52_out)

    data = torch.cat((onehot_out, blo_out, zscale_out, bin51_out,bin52_out), dim=2)

    data = np.array(list(data))
    label = np.array(list(label))

    # valid_pred = []
    # valid_label = []
    #
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    #
    # for index, (train_idx, val_idx) in enumerate(skf.split(data, label)):
    #     print('**' * 10, 'the', index + 1, 'fold', 'ing....', '**' * 10)
    #     x_train, x_valid = data[train_idx], data[val_idx]
    #     y_train, y_valid = label[train_idx], label[val_idx]
    #
    #     train_dataset = myDataset(x_train, y_train)
    #     valid_dataset = myDataset(x_valid, y_valid)
    #
    #     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0)
    #     valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, drop_last=True, num_workers=0)
    #
    #     model = MultiScaleConvNet()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-5)
    #     criterion = nn.BCELoss()
    #     best_val_score = float('-inf')
    #     last_improve = 0
    #     best_model = None
    #
    #     for epoch in range(20):
    #         print('epoch:', epoch)
    #         train_score = fit(model, train_loader, optimizer, criterion)
    #         val_score, _, _ = validate(model, valid_loader)
    #
    #         if val_score > best_val_score:
    #             best_val_score = val_score
    #             best_model = copy.deepcopy(model)
    #             last_improve = epoch
    #             improve = '*'
    #         else:
    #             improve = ''
    #
    #         print(f'Epoch: {epoch} Train Score: {train_score}, Valid Score: {val_score}  ')
    #     model = best_model
    #
    #     print(f"=============end!!!!================")
    #     print("train")
    #     train_score, _, _ = validate(model, train_loader)
    #     print("valid")
    #     valid_score, pred_list, label_list = validate(model, valid_loader)
    #     valid_pred.extend(pred_list)
    #     valid_label.extend(label_list)
    #
    # print("*****************************************5-fold cross valid**********************************************")
    #
    # print("cross_valid_score")
    # cross_valid_score = cal_score(valid_pred, valid_label)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=100)

    # 创建训练集和测试集的数据加载器
    train_dataset = myDataset(x_train, y_train)
    test_dataset = myDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=True)

    model = MultiScaleConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-05)
    criterion = nn.BCELoss()

    # 训练和测试模型
    for epoch in range(100):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = torch.tensor(data).to(device)
            target = torch.tensor(target).to(device)
            # 前向传播
            output = model(data)
            # 计算损失
            # L1正则化
            loss = loss_fn(output, target, model, 0.001)
            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sn, sp, mcc, acc, auc = test(model, test_loader, device)
        print('epoch {} ,sn: {:.4f}% ,sp: {:.4f}% ,mcc: {:.4f}% ,acc: {:.4f}% ,auc: {:.4f}% '
              .format(epoch, 100 * sn, 100 * sp, 100 * mcc, 100 * acc, 100 * auc))

    # # 训练和测试模型
    # for epoch in range(100):
    #     model.train()
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         data = torch.tensor(data).to(device)
    #         target = torch.tensor(target).to(device)
    #         # 前向传播
    #         output = model(data)
    #         # 计算损失
    #         # L1正则化
    #         loss = loss_fn(output, target, model, 0.0001)
    #         # 反向传播和参数更新
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     # 每个epoch结束之后，在测试集上计算损失和准确率
    #     model.eval()
    #     test_loss = 0
    #     correct = 0
    #     with torch.no_grad():
    #         for data, target in test_loader:
    #             data = torch.tensor(data).to(device)
    #             target = torch.tensor(target).to(device)
    #             # 前向传播
    #             output = model(data)
    #             # 计算损失
    #             output = output.squeeze()
    #             test_loss += criterion(output, target).item()
    #             # 统计正确分类的样本数
    #             output = output.unsqueeze(1)
    #             pred = (output > 0.5)
    #             pred = torch.tensor(pred)
    #             # pred = output.argmax(dim=1, keepdim=True)
    #             correct += pred.eq(target.view_as(pred)).sum().item()
    #
    #     # 输出本轮训练的损失和测试的准确率
    #     test_loss /= len(test_loader.dataset)
    #     test_acc = correct / len(test_loader.dataset)
    #     if test_acc > 0.95:
    #         torch.save(model, './best_model/' + 'model_' + str(epoch) + str(test_acc) + '.pth')
    #         print('save' + str(epoch + 1))
    #     print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}%, correct_num: {}'
    #           .format(epoch + 1, 20, loss.item(), test_loss, 100 * test_acc, correct))