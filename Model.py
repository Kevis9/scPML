import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from utils import lossPolt
'''
    CPM-Nets, 改写为Pytroch形式
'''

class CPMNets():
    def __init__(self, view_num, train_len, test_len, view_feat, lsd_dim=128):
        '''
        :param view_num:
        :param train_Len: training data length
        :param test_Len:
        :param view_feat: 一个数组，每一个view的特征长度
        :param lsd_dim:

        这里也不考虑用源码中的sn矩阵，因为我们不缺view，也没必要随机取产生缺失的view
        '''
        super(CPMNets, self).__init__()
        self.view_num = view_num
        self.view_feat = view_feat
        self.view_idx = [[] for v in self.view_feat] # 记录每一个view在data中的index
        cnt = 0
        for i in range(len(view_feat)):
            for j in range(view_feat[i]):
                self.view_idx[i].append(cnt)
                cnt+=1

        self.lsd_dim = lsd_dim
        self.train_len = train_len
        self.test_len = test_len

        self.h_train = torch.zeros((self.train_len, self.lsd_dim), requires_grad=True, dtype=torch.float)
        self.h_test = torch.zeros((self.test_len, self.lsd_dim), requires_grad=True, dtype=torch.float)

        # 初始化
        nn.init.xavier_uniform_(self.h_train)
        nn.init.xavier_uniform_(self.h_test)


        # 模型的搭建
        # net的输入是representation h ---> 重构成 X

        self.net = dict()
        for v_num in range(view_num):
            self.net[str(v_num)] = nn.Sequential(
                nn.Linear(self.lsd_dim, self.view_feat[v_num]), #我对源码的理解就是只有一层全连接
                # nn.Dropout(0.2)
            )


    def reconstrution_loss(self, r_x, x):
        '''
        :param r_x: 由 h 重构出来的x
        :param x:  原x
        :return: 返回 (r_x-x)^2
        '''
        return ((r_x - x)**2).sum()

    def classification_loss(self, gt):
        '''
        :param gt: ground truth labels (把train部分的label传进来),
        :return: 返回classfication的loss
        '''
        F_h_h = torch.mm(self.h_train, self.h_train.t())
        F_hn_hn = torch.diag(F_h_h)
        F_h_h = F_h_h - torch.diag_embed(F_hn_hn) # 将F_h_h对角线部分置0
        classes = torch.max(gt).item() - torch.min(gt).item() + 1
        label_onehot = torch.zeros((self.train_len, classes))
        gt = gt - 1 # 因为这里我的labels是从1开始的，矩阵从0开始，减1避免越界
        label_onehot.scatter_(dim=1, index=gt.view(-1,1), value=1) #得到各个样本分类的one-hot表示
        label_num = torch.sum(label_onehot, dim=0) #得到每个label的样本数
        F_h_h_sum = torch.mm(F_h_h, label_onehot)
        F_h_h_mean = F_h_h_sum / label_num #自动广播
        gt_ = torch.argmax(F_h_h_mean, dim=1) + 1 # 获得每个样本预测的类别
        F_h_h_mean_max = torch.max(F_h_h_mean, dim=1)[0] # 取到每个样本的最大值 1*n
        theta = torch.not_equal(gt, gt_).view(1,-1)
        F_h_hn_mean_ = torch.mul(F_h_h_mean, label_onehot)
        F_h_hn_mean = torch.sum(F_h_hn_mean_, dim=1) # 1*n

        return torch.sum(F.relu(theta + (F_h_h_mean_max - F_h_hn_mean)))


    def train_model(self, data, labels, n_epoch, lr=[0.001, 0.001]):
        '''
        这个函数直接对模型进行训练
        随着迭代，更新两个部分: net和h
        :param data: training data , type: Tensor , 行代表cell，列代表的是view
        :param labels: training labels
        :param n_epoch:
        :param lr: 学习率，是一个数组，0 for net， 1 for h
        :return:
        '''
        # 优化器
        netParams = []
        for v in range(self.view_num):
            for p in self.net[str(v)].parameters():
                netParams.append(p)

        optimizer_for_net = optim.Adam(params=netParams, lr=lr[0])
        optimizer_for_h = optim.Adam(params=[self.h_train], lr=lr[1])
        r_loss_list = []
        c_loss_list = []
        for epoch in range(n_epoch):
            r_loss = 0
            for v in range(self.view_num):
                r_loss += self.reconstrution_loss(self.net[str(v)](self.h_train), data[:, self.view_idx[v]])
            c_loss = self.classification_loss(labels)
            all_loss = r_loss + c_loss

            optimizer_for_net.zero_grad()
            optimizer_for_h.zero_grad()

            all_loss.backward() #这里的想法是这样：对net求导，另外reconstruction的loss就没了，对h求导，两边都要考虑
            # 更新net部分
            optimizer_for_net.step()
            # 更新h部分
            optimizer_for_h.step()

            # 这里应该打印平均的loss（也就是每一个样本的复原的loss）
            if epoch%1000 == 0:
                print('epoch %d: Reconstruction loss = %.3f, classification loss = %.3f'%(epoch, r_loss.detach().item()/self.train_len, c_loss.detach().item()/self.train_len))
            r_loss_list.append(r_loss.detach().item()/self.train_len)
            c_loss_list.append(c_loss.detach().item()/self.train_len)
        lossPolt(r_loss_list, c_loss_list, n_epoch)

    def get_h_train(self):
        return self.h_train

    def get_h_test(self):
        return self.h_test


class scGNN(torch.nn.Module):
    def __init__(self, G_data):
        super(scGNN, self).__init__()
        #Bottle-necked
        # middle_out = int(max(5, G_data.num_features/64))
        middle_out = int(max(8, G_data.num_features / 2))
        self.conv1 = GCNConv(G_data.num_features, middle_out)
        self.conv2 = GCNConv(middle_out, G_data.num_features)

    def forward(self, G_data):
        x, edge_index = G_data.x, G_data.edge_index
        # 中间夹了一层relu和一层dropout(避免过拟合的发生)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    def get_embedding(self, G_data):
        x, edge_index = G_data.x, G_data.edge_index
        return self.conv1(x, edge_index)



