import numpy as np
import torch.nn as nn
import torch
import wandb
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from utils import cpm_classify
'''
    CPM-Nets, 改写为Pytroch形式
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CPMNets():
    def __init__(self, view_num, train_len, test_len, view_d_arr, class_num, lsd_dim, w):
        '''
        :param view_num: view的数目
        :param train_len: training data length
        :param test_len: test data length
        :param view_d_arr: 一个数组，代表每一个view的特征长度
        :param lsd_dim: latent space H dimension
        :param class_num: 类别数
        :param w: classfication loss的权重
        这里也不考虑用源码中的sn矩阵，因为我们不缺view，也没必要随机取产生缺失的view
        '''
        super(CPMNets, self).__init__()
        self.w = w
        self.view_num = view_num
        # 记录每一个view在data中的index 比如第一个view的长度是10，那么0,1,...,9都放在view_idx[0]中
        self.view_idx = [[] for v in view_d_arr]
        cnt = 0
        for i in range(len(view_d_arr)):
            for j in range(view_d_arr[i]):
                self.view_idx[i].append(cnt)
                cnt += 1

        self.lsd_dim = lsd_dim
        self.train_len = train_len
        self.test_len = test_len

        self.h_train = torch.zeros((self.train_len, self.lsd_dim), dtype=torch.float).to(device)
        self.h_train.requires_grad = True # 先放在GPU上再设置requires_grad

        self.h_test = torch.zeros((self.test_len, self.lsd_dim), dtype=torch.float).to(device)
        self.h_test.requires_grad = True

        self.class_num = class_num

        # 初始化
        nn.init.xavier_uniform_(self.h_train)
        nn.init.xavier_uniform_(self.h_test)

        # 模型的搭建
        # net的输入是representation h ---> 重构成 X
        self.net = dict()
        for i in range(view_num):
            self.net[str(i)] = nn.Sequential(
                nn.Linear(self.lsd_dim, view_d_arr[i], device=device),  # 我对源码的理解就是只有一层全连接
                # nn.ReLU(),
                # nn.Linear(2 * self.lsd_dim,  view_d_arr[i])
                # nn.Dropout(0.2)
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
        # classes = torch.max(gt).item() - torch.min(gt).item() + 1   # class数量
        label_onehot = torch.zeros((h.shape[0], self.class_num)).to(device)
        # gt = gt - 1  # 因为这里我的labels是从1开始的，矩阵从0开始，减1避免越界

        label_onehot.scatter_(dim=1, index=gt.view(-1, 1), value=1)  # 得到各个样本分类的one-hot表示
        label_num = torch.sum(label_onehot, dim=0)  # 得到每个label的样本数
        F_h_h_sum = torch.mm(F_h_h, label_onehot)

        # print('From classification_loss, label num is : {:}'.format(label_num))

        label_num[torch.where(label_num==0)] = 1 # 这里要排除掉为分母为0的风险(transfer across species里面有这种情况)
        
        F_h_h_mean = F_h_h_sum / label_num  # 自动广播
        gt_ = torch.argmax(F_h_h_mean, dim=1)  # 获得每个样本预测的类别
        F_h_h_mean_max = torch.max(F_h_h_mean, dim=1)[0]  # 取到每个样本的最大值 1*n
        theta = torch.not_equal(gt, gt_).view(1, -1)
        F_h_hn_mean_ = torch.mul(F_h_h_mean, label_onehot)
        F_h_hn_mean = torch.sum(F_h_hn_mean_, dim=1)  # 1*n

        return torch.sum(F.relu(theta + (F_h_h_mean_max - F_h_hn_mean)))
        # return torch.sum(F.relu(F_h_h_mean_max - F_h_hn_mean))

    def fisher_loss(self, gt):
        '''
        给出LDA中fisher_loss
        :param gt:
        :return:
        '''
        label_list = list(set(np.array(gt.cpu()).reshape(-1).tolist()))
        idx = [] # 存储每一个类别所在行数

        for label in label_list:
            idx.append(torch.where(gt==label)[1])

        v_arr = []
        u_arr = []
        for i in range(len(idx)):

            data = self.h_train[idx[i], :]

            u = torch.mean(data, dim=0, dtype=torch.float64)

            v_arr.append((torch.diag(torch.mm(data-u, (data-u).T)).sum()/(data.shape[0])).view(1,-1))
            u_arr.append(u.reshape(1, -1))

        variance_loss = torch.cat(v_arr, dim=1).sum()


        u_tensor = torch.cat(u_arr, dim=0)
        u_tensor = torch.mm(u_tensor, u_tensor.T)

        # 将对角线置为0
        u_diag = torch.diag(u_tensor)
        u_tensor = u_tensor - torch.diag_embed(u_diag)

        # 簇之间的距离
        dist_loss = u_tensor.sum()

        return F.relu(variance_loss - dist_loss)

    def similarity_loss(self, ref_h, query_h):
        '''
        对于paired omics data，我们计算每个样本feature的Similarity
        :param ref_h:
        :param query_h:
        :return:
        '''
        ref_h = ref_h.to(device)
        query_h = query_h.to(device)
        F_ref_query = torch.mm(ref_h, query_h.t())
        F_diag = torch.diag(F_ref_query)
        return -torch.sum(F_diag)

    def train_ref_h(self, data, labels, n_epochs, lr):
        '''
        这个函数直接对模型进行训练
        随着迭代，更新两个部分: net参数和h_train
        :param data: training data , type: Tensor , (cell * views)
        :param labels: training labels
        :param n_epochs: epochs
        :param lr: 学习率，是一个数组，0 for net， 1 for h
        :return:
        '''
        data = data.to(device)
        labels = labels.to(device)
        # 优化器
        netParams = []
        for v in range(self.view_num):
            for p in self.net[str(v)].parameters():
                netParams.append(p)

        optimizer_for_net = optim.Adam(params=netParams, lr=lr[0])
        optimizer_for_h = optim.Adam(params=[self.h_train], lr=lr[1])

        for epoch in range(n_epochs):
            r_loss = 0
            for i in range(self.view_num):
                r_loss += self.reconstrution_loss(self.net[str(i)](self.h_train), data[:, self.view_idx[i]])

            # 每个样本的平均loss
            # r_loss = r_loss / self.train_len

            c_loss = self.classification_loss(self.h, labels)


            # 每个样本的平均loss, 在这里 *w 来着重降低 classfication loss
            all_loss = r_loss + self.w * c_loss

            optimizer_for_net.zero_grad()
            optimizer_for_h.zero_grad()

            # 这里的写all_loss想法是这样：对net参数求导，c_loss为0，对h求导，r_loss和c_loss都要考虑
            all_loss.backward()
            # 更新net部分
            optimizer_for_net.step()
            # 更新h部分
            optimizer_for_h.step()

            # 这里应该打印平均的loss（也就是每一个样本的复原的loss）
            if epoch % 1000 == 0:
                print('epoch %d: Reconstruction loss = %.3f, classification loss = %.3f' % (
                    epoch, r_loss.detach().item(), c_loss.detach().item()))
            wandb.log({
                'CPM train: reconstruction loss': r_loss.detach().item(),
                'CPM train: classification loss': c_loss.detach().item()
            })

    def train_query_h(self, data, n_epochs, do_omics):
        '''
        :param data: query data
        :param n_epochs:
        :param do_omics: Bool: True for the omics part.
        :return:
        '''
        data = data.to(device)
        optimizer_for_query_h = optim.Adam(params=[self.h_test])

        for epoch in range(n_epochs):
            r_loss = 0
            for v in range(self.view_num):
                r_loss += self.reconstrution_loss(self.net[str(v)](self.h_test), data[:, self.view_idx[v]])

            all_loss = r_loss
            if do_omics:
                # similarity loss
                s_loss = self.similarity_loss(self.h_train, self.h_test)
                all_loss = F.relu(all_loss + s_loss)


            optimizer_for_query_h.zero_grad()
            all_loss.backward()
            optimizer_for_query_h.step()


            if epoch % 1000 == 0:
                print('Train query h: epoch %d: Reconstruction loss = %.3f'%(
                    epoch, r_loss.detach().item()), end=" ")
                if do_omics:
                    print('Similarity loss = %.3f' % (
                        s_loss.detach().item()))
                else:
                    print("")

            wandb.log({
                'CPM query h: reconstruction loss': r_loss.detach().item()
            })
            if do_omics:
                wandb.log({
                    'CPM query h: similarity loss': s_loss.detach().item()
                })


    def get_h_train(self):
        return self.h_train

    def get_h_test(self):
        return self.h_test


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
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_embedding(self, G_data):
        x, edge_index = G_data.x, G_data.edge_index
        return self.conv1(x.to(device), edge_index.to(device))


class Classifier(nn.Module):
    def __init__(self, lsd, class_num):
        '''
            lsd: latent space dimentionlity            
        '''
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(lsd, int(lsd/2)),
            nn.ReLU()
        )
        self.prediction_layer = nn.Sequential(            
            nn.Linear(int(lsd/2), class_num)
        )
            
    def forward(self, x):
        x = self.embedding_layer(x)
        logits = self.prediction_layer(x)
        return logits
    
    def get_embedding(self, x):
        return self.embedding_layer(x)
    