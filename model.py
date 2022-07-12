import numpy as np
import torch.nn as nn
import torch
import wandb
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import pdist
from utils import cpm_classify
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

'''
    CPM-Nets, 改写为Pytroch形式
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiViewDataSet(Dataset):
    def __init__(self, multi_view_data, labels, latent_representation, view_num):
        '''
        :param multi_view_data: list类型，保存每个view ，每个元素都是tensor
        :param labels:
        :param latent_representation: 相当于h
        :param view_dim_arr:
        '''
        super(MultiViewDataSet, self).__init__()
        self.multi_view_data = multi_view_data
        self.labels = labels.view(-1)
        self.latent_representation = latent_representation
        self.view_num = view_num

    def __getitem__(self, item):
        data = []
        for i in range(self.view_num):
            data.append(self.multi_view_data[i][item].view(1, -1))
        if self.labels is not None:
            return torch.concat(data, dim=1).view(-1), self.latent_representation[item], self.labels[item]
        else:
            return torch.concat(data, dim=1).view(-1), self.latent_representation[item]

    def __len__(self):
        return self.multi_view_data[0].shape[0]

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        '''
        :param gamma: (1-pt)**gamma
        :param alpha:  权重list
        :param class_num:
        '''
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if not isinstance(alpha, torch.Tensor):
            self.alpha = torch.FloatTensor(alpha).to(device)
        else:
            self.alpha = alpha

    def forward(self, logits, gt):
        # 先做一个softmax归一化， 然后计算log，这样得到了 log(p)
        preds_logsoft = F.log_softmax(logits, dim=1)
        preds_softmax = torch.exp(preds_logsoft)    # 这里对上面的log做一次exp，得到只计算softmax的数值

        preds_logsoft = preds_logsoft.gather(1, gt.view(-1, 1))
        preds_softmax = preds_softmax.gather(1, gt.view(-1, 1))
        self.alpha = self.alpha.gather(0, gt.view(-1))

        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        return loss.mean()


class CPMNets(torch.nn.Module):
    def __init__(self, view_num, train_len, view_dim_arr, ref_labels, config):
        super(CPMNets, self).__init__()
        self.config = config
        self.view_num = view_num
        # 记录每一个view在data中的index 比如第一个view的长度是10，那么0,1,...,9都放在view_idx[0]中
        self.view_idx = [[] for v in view_dim_arr]
        cnt = 0
        for i in range(len(view_dim_arr)):
            for j in range(view_dim_arr[i]):
                self.view_idx[i].append(cnt)
                cnt += 1

        self.lsd_dim = config['lsd_dim']
        self.train_len = train_len

        self.h_train = torch.zeros((self.train_len, self.lsd_dim), dtype=torch.float).to(device)
        self.h_train.requires_grad = True  # 先放在GPU上再设置requires_grad
        # self.class_num = config['ref_class_num']

        self.ref_label_name = ref_labels
        # 初始化
        nn.init.xavier_uniform_(self.h_train)

        # 模型的搭建
        # net的输入是representation h ---> 重构成 X
        self.net = dict()
        for i in range(view_num):
            self.net[str(i)] = nn.Sequential(
                nn.Linear(self.lsd_dim, view_dim_arr[i], device=device),

            )

        self.classnum = len(set(ref_labels))
        self.classifier = nn.Sequential(
            nn.Linear(self.lsd_dim, self.classnum, device=device),
        )

    def reconstrution_loss(self, r_x, x):
        '''
        :param r_x: 由 h 重构出来的x
        :param x:  原x
        :return: 返回 (r_x-x)^2
        '''
        return ((r_x - x) ** 2).sum()

    def classification_loss(self, h, gt):
        '''
        :param gt: ground truth labels (把train部分的label传进来),

        :param h: latent representation
        :param len:
        :return: 返回classfication的loss
        '''
        F_h_h = torch.mm(h, h.t())
        F_hn_hn = torch.diag(F_h_h)
        F_h_h = F_h_h - torch.diag_embed(F_hn_hn)  # 将F_h_h对角线部分置0
        class_num = torch.max(gt).item() - torch.min(gt).item() + 1   # class数量
        # label_onehot = torch.zeros((h.shape[0], self.class_num)).to(device)

        label_onehot = torch.zeros((h.shape[0], class_num)).to(device)
        # gt = gt - 1  # 因为这里我的labels是从1开始的，矩阵从0开始，减1避免越界

        label_onehot.scatter_(dim=1, index=gt.view(-1, 1), value=1)  # 得到各个样本分类的one-hot表示
        label_num = torch.sum(label_onehot, dim=0)  # 得到每个label的样本数
        F_h_h_sum = torch.mm(F_h_h, label_onehot)
        label_num[torch.where(label_num == 0)] = 1  # 这里要排除掉为分母为0的风险(transfer across species里面有这种情况)

        F_h_h_mean = F_h_h_sum / label_num  # 自动广播
        gt_ = torch.argmax(F_h_h_mean, dim=1)  # 获得每个样本预测的类别
        F_h_h_mean_max = torch.max(F_h_h_mean, dim=1)[0]  # 取到每个样本的最大值 1*n
        theta = torch.not_equal(gt, gt_).view(1, -1)
        F_h_hn_mean_ = torch.mul(F_h_h_mean, label_onehot)
        F_h_hn_mean = torch.sum(F_h_hn_mean_, dim=1)  # 1*n

        return torch.sum(F.relu(theta + (F_h_h_mean_max - F_h_hn_mean)))

    def train_ref_h(self, multi_view_data, labels):

        '''
        这个函数直接对模型进行训练
        随着迭代，更新两个部分: net参数和h_train
        :param multi_view_data: training data , type: Tensor , (cell * views)
        :param labels: training labels
        :param n_epochs: epochs
        :param lr: 学习率，是一个数组，0 for net， 1 for h
        :return:
        '''
        # multi_view_data = multi_view_data.to(device)
        # labels = labels.to(device)
        # 优化器
        netParams = []
        for v in range(self.view_num):
            for p in self.net[str(v)].parameters():
                netParams.append(p)

        optimizer_for_net = optim.Adam(params=netParams)
        optimizer_for_h = optim.Adam(params=[self.h_train])
        optimizer_for_classifier = optim.Adam(params=self.classifier.parameters())
        # criterion = nn.CrossEntropyLoss()   #暂时考虑用CrossEntropy，如果样本不太均衡就用Focal loss
        # 确定各类别的比例，用 (1-x) / (1-x).sum() 归一
        alpha = torch.unique(labels, return_counts=True)[1]
        alpha = alpha / alpha.sum()

        alpha = (1-alpha) / (1-alpha).sum()
        # Gamma = 0退化为带平衡因子的CE
        criterion = FocalLoss(gamma=0, alpha=alpha)
        # 数据准备
        dataset = MultiViewDataSet(multi_view_data, labels, self.h_train, self.view_num)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.config['batch_size_cpm'])
        labels = labels.view(-1)
        for epoch in range(self.config['epoch_CPM_train']):

            # 更新reconstruction net
            for batch in dataloader:
                b_data, b_h, _ = batch
                b_data = b_data.to(device)
                b_h = b_h.to(device)
                r_loss = 0
                for i in range(self.view_num):
                    r_loss += self.reconstrution_loss(self.net[str(i)](b_h), b_data[:, self.view_idx[i]])
                optimizer_for_net.zero_grad()
                r_loss.backward()
                optimizer_for_net.step()

            # 更新classifier
            for batch in dataloader:
                b_data, b_h, b_label = batch
                b_data = b_data.to(device)
                b_h = b_h.to(device)
                b_label = b_label.to(device)

                logits = self.classifier(b_h)
                c_loss = criterion(logits, b_label)
                optimizer_for_classifier.zero_grad()
                c_loss.backward()
                optimizer_for_classifier.step()

            # 更新h
            # 这里没有选择用batch的方式去更新h，觉得h在这里作为参数，不适合用batch的方式去更新
            for step in range(10):
                data_tensor = torch.concat(multi_view_data, dim=1)
                r_loss = 0
                for i in range(self.view_num):
                    r_loss += self.reconstrution_loss(self.net[str(i)](self.h_train), data_tensor[:, self.view_idx[i]])
                logits = self.classifier(self.h_train)

                c_loss = criterion(logits, labels)

                total_loss = r_loss + self.config['w_classify'] * c_loss
                optimizer_for_h.zero_grad()
                total_loss.backward()
                optimizer_for_h.step()


            print('epoch %d: Reconstruction loss = %.3f, classification loss = %.3f' % (
                epoch, r_loss.detach().item(), c_loss.detach().item()))
            wandb.log({
                'CPM train: reconstruction loss': r_loss.detach().item(),
                'CPM train: classification loss': c_loss.detach().item(),
                # 'CPM train: fisher loss': f_loss.detach().item()
                # 'CPM train: center loss': cen_loss.detach().item()
            })

    def train_query_h(self, data, n_epochs, labels):
        '''
        :param data: query data, not a list
        :param n_epochs:
        :param labels, 用来测试acc，选择最好的
        :return:
        '''
        data = data.to(device)
        labels = labels.to(device)
        h_test = torch.zeros((data.shape[0], self.lsd_dim), dtype=torch.float).to(device)
        h_test.requires_grad = True
        nn.init.xavier_uniform_(h_test)
        optimizer_for_query_h = optim.Adam(params=[h_test])

        # 变成eval模式，不更新参数，以及可以去掉dropout层的计算
        for v in range(self.view_num):
            self.net[str(v)].eval()

        # 数据准备
        # dataset = MultiViewDataSet(multi_view_data, None, h_test, self.view_num)
        # dataloader = DataLoader(dataset, shuffle=False, batch_size=128)
        max_acc = 0
        max_iter = 0
        for epoch in range(n_epochs):
            r_loss = 0
            for v in range(self.view_num):
                r_loss += self.reconstrution_loss(self.net[str(v)](h_test), data[:, self.view_idx[v]])
            optimizer_for_query_h.zero_grad()
            r_loss.backward()
            optimizer_for_query_h.step()
            acc = self.evaluate(h_test, labels)
            if max_acc < acc:
                max_acc = acc
                max_iter = epoch

            print('Train query h: epoch %d: Reconstruction loss = %.3f' % (epoch, r_loss.detach().item()))

            wandb.log({
                'CPM query h: reconstruction loss': r_loss.detach().item()
            })
        print("Max Acc is {:.3f}, epoch: {:}".format(max_acc, max_iter))

        return h_test

    def evaluate(self, val_data, val_labels):
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(val_data)
            pred = logits.argmax(dim=1)

            acc = (pred == val_labels).sum().item() / val_data.shape[0]
            return acc

    def classify(self, data):
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(data)
            pred = logits.argmax(dim=1)
            return pred

    def get_h_train(self):
        return self.h_train

    def get_ref_labels(self):
        return self.ref_label_name

    def forward(self):
        pass


class scGNN(torch.nn.Module):
    def __init__(self, G_data, middle_out):
        super(scGNN, self).__init__()
        # Bottle-necked
        # middle_out = int(max(5, G_data.num_features/64))
        # middle_out = int(max(8, G_data.num_features / 2))
        # middle_out = 128
        self.conv1 = GCNConv(G_data.num_features, middle_out)
        self.conv2 = GCNConv(middle_out, G_data.num_features)

    def forward(self, G_data):
        x, edge_index = G_data.x, G_data.edge_index
        # 中间夹了一层relu和一层dropout(避免过拟合的发生)
        x = F.relu(self.conv1(x, edge_index))
        # 可以调整dropout的比率
        x = F.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def get_embedding(self, G_data):
        x, edge_index = G_data.x, G_data.edge_index
        return self.conv1(x.to(device), edge_index.to(device))
