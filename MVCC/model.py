import os.path

import numpy as np
import torch.nn as nn
import torch
import wandb
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import pdist
from MVCC.util import cpm_classify, mask_data, construct_graph, z_score_scale
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from torch.utils.data import Dataset, TensorDataset

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
        preds_softmax = torch.exp(preds_logsoft)  # 这里对上面的log做一次exp，得到只计算softmax的数值

        preds_logsoft = preds_logsoft.gather(1, gt.view(-1, 1))
        preds_softmax = preds_softmax.gather(1, gt.view(-1, 1))
        self.alpha = self.alpha.gather(0, gt.view(-1))

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        return loss.mean()


class CPMNets(torch.nn.Module):
    def __init__(self, view_num, view_dim, lsd, class_num, save_path, lamb):
        '''
        :param view_num:
        :param view_dim: 每个view的维度，在这里我们的view的维度一致
        :param lsd: latent space dimension
        :param class_num: 类别数
        '''
        super(CPMNets, self).__init__()

        self.view_num = view_num
        self.lsd = lsd
        self.view_dim = view_dim
        self.save_path = save_path
        self.lamb = lamb
        # net的输入是representation h ---> 重构成 X
        # 重构的网络
        self.net = []
        self.ref_h = None
        self.ref_labels = None
        self.query_h = None
        for i in range(view_num):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.lsd, self.view_dim, device=device),
                )
            )

        # 分类的网络
        self.class_num = class_num
        self.classifier = nn.Sequential(
            nn.Linear(self.lsd, self.class_num, device=device),
            nn.Dropout(0.05)
        )

    def reconstrution_loss(self, r_x, x):
        '''
        :param r_x: 由 h 重构出来的x
        :param x:  原x
        :return: 返回 (r_x-x)^2
        '''
        return ((r_x - x) ** 2).sum()

    def fisher_loss(self, train_h, gt):
        '''
        给出LDA中fisher_loss
        :param gt:
        :return:
        '''
        label_list = list(set(np.array(gt.detach().cpu()).reshape(-1).tolist()))
        idx = []  # 存储每一个类别所在行数

        for label in label_list:
            idx.append(torch.where(gt == label)[0])

        v_arr = []
        u_arr = []
        for i in range(len(idx)):
            data = train_h[idx[i], :]

            u = torch.mean(data, dim=0, dtype=torch.float64)

            v_arr.append((torch.diag(torch.mm(data - u, (data - u).T)).sum() / (data.shape[0])).view(1, -1))
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

    def class_loss(self, h, gt):
        '''
        :param gt: ground truth labels (把train部分的label传进来),

        :param h: latent representation
        :param len:
        :return: 返回classfication的loss
        '''
        F_h_h = torch.mm(h, h.t())
        F_hn_hn = torch.diag(F_h_h)
        F_h_h = F_h_h - torch.diag_embed(F_hn_hn)  # 将F_h_h对角线部分置0
        class_num = torch.max(gt).item() - torch.min(gt).item() + 1  # class数量
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

    def train_ref_h(self, ref_data, labels, batch_size, epochs, patience, early_stop):

        '''
        这个函数直接对模型进行训练
        随着迭代，更新net，h以及classifier
        在更新h的时候，可以考虑加入原论文的classification loss
        :param ref_data: training data, ndarray, not a list
        :param labels:
        :param n_epochs: epochs
        :param lr: 学习率，是一个数组，0 for net， 1 for h
        :return:
        '''
        # 优化器
        net_params = []
        for i in range(self.view_num):
            for p in self.net[i].parameters():
                net_params.append(p)

        # 将数据分为train和valid data
        labels = labels.view(-1)

        train_data = ref_data
        train_label = labels
        # 只为train data创建h, val h由后面的评估生成
        train_h = torch.zeros((train_data.shape[0], self.lsd), dtype=torch.float).to(device)
        train_h.requires_grad = True  # 先放在GPU上再设置requires_grad
        nn.init.xavier_uniform_(train_h)

        optimizer_for_net = optim.Adam(params=net_params)
        optimizer_for_train_h = optim.Adam(params=[train_h])
        optimizer_for_classifier = optim.Adam(params=self.classifier.parameters())

        # 确定各类别的比例，用 (1-x) / (1-x).sum() 归一
        alpha = torch.unique(labels, return_counts=True)[1]
        alpha = alpha / alpha.sum()
        alpha = (1 - alpha) / (1 - alpha).sum()
        # gamma=0, 退化为带平衡因子的CE
        # 这个Focal loss在pltaform task1里面带来了很大的作用
        criterion = FocalLoss(gamma=1, alpha=alpha)
        # criterion = nn.CrossEntropyLoss()
        # 数据准备
        train_data_arr = [train_data[:, i * self.view_dim: (i + 1) * self.view_dim] for i in range(self.view_num)]
        dataset_train = MultiViewDataSet(train_data_arr, train_label, train_h, self.view_num)
        dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
        train_label = train_label.view(-1)

        # 利用Silhouette score来做早停处理
        min_c_loss = 99999999999
        stop = 0

        # 设置model为train模式
        for i in range(self.view_num):
            self.net[i].train()

        for epoch in range(epochs):
            # 更新net
            train_h.requires_grad = False
            for i in range(self.view_num):
                for param in self.net[i].parameters():
                    param.requires_grad = True

            r_loss = 0
            for i in range(self.view_num):
                r_loss += self.reconstrution_loss(self.net[i](train_h),
                                                  train_data[:, i * self.view_dim: (i + 1) * self.view_dim])
            # r_loss /= train_h.shape[0]
            optimizer_for_net.zero_grad()
            r_loss.backward()
            optimizer_for_net.step()

            # 更新h
            train_h.requires_grad = True
            for i in range(self.view_num):
                for param in self.net[i].parameters():
                    param.requires_grad = False
            r_loss = 0
            for i in range(self.view_num):
                r_loss += self.reconstrution_loss(self.net[i](train_h),
                                                  train_data[:, i * self.view_dim: (i + 1) * self.view_dim])

            c_loss = self.class_loss(train_h, train_label)
            total_loss = r_loss + self.lamb * c_loss
            optimizer_for_train_h.zero_grad()
            total_loss.backward()
            optimizer_for_train_h.step()

            if not early_stop:
                if epoch % 20 == 0:
                    print(
                        'epoch %d: Reconstruction loss = %.3f, classification loss = %.3f.' % (
                            epoch, r_loss.detach().item(), c_loss.detach().item()))
            else:

                if c_loss < min_c_loss:
                    min_c_loss = c_loss
                    best_train_h = train_h.detach().clone()
                    torch.save(self, os.path.join(self.save_path, 'cpm_model.pt'))
                    stop = 0
                    print(
                        'epoch {:}: Reconstruction loss = {:.3f}, classification loss = {:.3f}. save cpm model'.format(
                            epoch, r_loss.detach().item(), c_loss.detach().item()))
                else:
                    stop += 1
                    if stop > patience:
                        print("CPM train h stop at train epoch {:}, min classification loss is {:.3f}".format(epoch,
                                                                                                           min_c_loss))
                        break

        '''
            classifier拿出来单独更新
        '''
        if early_stop:
            # 如果之前用到了早停法，这里要重新加载这两个重要的模型
            best_cpm_model = torch.load(os.path.join(self.save_path, 'cpm_model.pt'))
            self.net = best_cpm_model.net
            train_h = best_train_h.detach().clone()

        train_h.requires_grad = False
        for i in range(self.view_num):
            for param in self.net[i].parameters():
                param.requires_grad = False

        print("Train  classifier")

        train_x, val_x, train_y, val_y = train_test_split(train_h.detach().cpu().numpy(),
                                                          train_label.detach().cpu().numpy(),
                                                          shuffle=True,
                                                          stratify=train_label.detach().cpu().numpy(),
                                                          test_size=0.2,
                                                          random_state=0
                                                          )
        train_x, val_x = torch.from_numpy(train_x).float().to(device), torch.from_numpy(val_x).float().to(device)
        train_y, val_y = torch.from_numpy(train_y).long().to(device), torch.from_numpy(val_y).long().to(device)

        dataset_train = TensorDataset(train_x, train_y)
        dataloader_train = DataLoader(dataset_train, shuffle=False, batch_size=batch_size)

        dataset_val = TensorDataset(val_x, val_y)
        dataloader_val = DataLoader(dataset_val, shuffle=False, batch_size=batch_size)

        # 更新classifier
        val_max_acc = 0
        stop = 0

        for epoch in range(epochs):
            # self.classifier.train()
            # logits = self.classifier(train_h)
            # c_loss = criterion(logits, train_label)
            # train_acc = (logits.argmax(dim=1) == train_label).sum().item() / train_label.shape[0]
            # optimizer_for_classifier.zero_grad()
            # c_loss.backward()
            # optimizer_for_classifier.step()

            c_loss_total = 0
            train_pred_arr = []
            train_true_label = []
            for batch in dataloader_train:
                b_h, b_label = batch
                b_h = b_h.to(device)
                b_label = b_label.to(device)
                logits = self.classifier(b_h)
                c_loss = criterion(logits, b_label)

                optimizer_for_classifier.zero_grad()
                c_loss.backward()
                optimizer_for_classifier.step()

                train_pred_arr.append(logits.detach().argmax(dim=1))
                train_true_label.append(b_label.detach())
                c_loss_total += c_loss.detach().item()

            preds = torch.concat(train_pred_arr)
            trues = torch.concat(train_true_label)
            train_acc = (preds == trues).sum() / trues.shape[0]
            '''
                早停处理 (这里只是为了训练出更好的classifier)
            '''
            val_acc = self.evaluate(dataloader_val)
            if early_stop:
                if val_max_acc < val_acc:
                    val_max_acc = val_acc
                    stop = 0

                    print(
                        'epoch {:}: train classification loss = {:.3f}, train acc is {:.3f}, val max acc is {:.3f}, save the model.'.format(
                            epoch, c_loss_total, train_acc, val_max_acc))

                    torch.save(self, os.path.join(self.save_path, 'cpm_model.pt'))
                else:
                    stop += 1
                    if stop > patience:
                        print("CPM train h stop at train epoch {:}, train acc is {:.3f}, val max acc is {:.3f}".format(
                            epoch, train_acc, val_max_acc))
                        break
            else:
                if epoch % 20 == 0:
                    print(
                        'epoch {:}: classification loss = {:.3f}. train acc is {:.3f}, val acc is {:.3f}'.format(
                            epoch, c_loss_total, train_acc, val_acc))
            # wandb.log({
            #     # 'CPM train: reconstruction loss': r_loss.detach().item(),
            #     'CPM train: classification loss': c_loss_total.detach().item(),
            # })
        self.ref_h = train_h
        self.ref_labels = train_label
        return train_h, train_label

    def train_query_h(self, data, n_epochs, early_stop, patience):
        '''
        :param data: query data, not a list
        :param n_epochs: epochs for reconstruction
        :return:
        '''
        data = data.to(device)

        h_test = torch.zeros((data.shape[0], self.lsd), dtype=torch.float).to(device)
        h_test.requires_grad = True
        nn.init.xavier_uniform_(h_test)
        optimizer_for_query_h = optim.Adam(params=[h_test])
        for i in range(self.view_num):
            self.net[i].eval()

        # 数据准备
        # dataset = MultiViewDataSet(multi_view_data, None, h_test, self.view_num)
        # dataloader = DataLoader(dataset, shuffle=False, batch_size=128)
        # 这里不确定什么时候的重构epoch是最好的，所以引入早停法，找到最大acc的epoch (修改：不再用早停，这个逻辑用在test部分有点牵强）
        min_r_loss = 99999999999
        min_r_loss_test_h = 0
        stop = 0
        for epoch in range(n_epochs):
            r_loss = 0
            for i in range(self.view_num):
                r_loss += self.reconstrution_loss(self.net[i](h_test),
                                                  data[:, i * self.view_dim: (i + 1) * self.view_dim])
            optimizer_for_query_h.zero_grad()
            r_loss.backward()
            optimizer_for_query_h.step()
            # acc = self.evaluate(h_test, labels)
            wandb.log({
                'CPM query h: reconstruction loss': r_loss.detach().item()
            })
            if not early_stop:
                if epoch % 20 == 0:
                    print('CPM query h: reconstruction loss {:}'.format(r_loss.detach().item()))
            else:
                if r_loss < min_r_loss:
                    min_r_loss = r_loss
                    min_r_loss_test_h = h_test.detach().clone()
                    stop = 0
                    if epoch%20 == 0:
                        print("Epoch {:}, train test h min reconstruction is {:.3f}, save test h".format(epoch, min_r_loss))
                else:
                    stop += 1
                    print("patience {:}/{:}".format(stop, patience))
                    if stop > patience:
                        print("Train test h, stop at epoch {:}".format(epoch))
                        break
        if not early_stop:
            min_r_loss_test_h = h_test
        # return h_test
        return min_r_loss_test_h

    def evaluate(self, dataloader_val):
        self.classifier.eval()
        with torch.no_grad():
            pred_list = []
            true_list = []
            for b_data, b_label in dataloader_val:
                logits = self.classifier(b_data)
                pred = logits.argmax(dim=1)
                pred_list.append(pred)
                true_list.append(b_label)
            preds = torch.concat(pred_list)
            trues = torch.concat(true_list)

            acc = (preds == trues).sum().item() / preds.shape[0]
            return acc

    def classify(self, h):
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(h)
            pred = logits.argmax(dim=1)
            return pred

    def forward(self, data):
        return self.classifier(data)


class scGNN(torch.nn.Module):
    def __init__(self, input_dim, middle_out):
        super(scGNN, self).__init__()
        # Bottle-necked
        # middle_out = int(max(5, G_data.num_features/64))
        # middle_out = int(max(8, G_data.num_features / 2))
        # middle_out = 128
        self.conv1 = GCNConv(input_dim, middle_out)
        self.conv2 = GCNConv(middle_out, input_dim)

    def forward(self, G_data):
        x, edge_index = G_data.x, G_data.edge_index
        # 中间夹了一层relu和一层dropout(避免过拟合的发生)
        x = F.relu(self.conv1(x, edge_index))
        # 可以调整dropout的比率
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_embedding(self, G_data):
        x, edge_index = G_data.x, G_data.edge_index
        return self.conv1(x.to(device), edge_index.to(device))


class MVCCModel:

    def __init__(self,
                 gcn_middle_out,
                 gcn_input_dim,
                 lsd,
                 class_num,
                 cpm_exist,
                 gcn_exist,
                 view_num=4,
                 epoch_gcn=3000,
                 k_neighbor=2,
                 epoch_cpm_train=500,
                 epoch_cpm_test=3000,
                 batch_size_cpm=256,
                 save_path='',
                 patience=50,
                 early_stop=True,
                 mask_rate=0.3,
                 patience_for_query = 500,
                 lamb=1):
        '''
        :param pretrained:
        :param early_stop:
        :param gcn_middle_out:
        :param view_num:
        :param epoch_gcn:
        :param k_neighbor:
        :param mask_rate:
        :param epoch_cpm_train:
        :param epoch_cpm_test:
        :param batch_size_cpm:
        :param lamb:
        :param val_size:
        '''
        self.lsd = lsd
        self.gcn_exist = gcn_exist
        self.cpm_exist = cpm_exist
        self.class_num = class_num
        self.gcn_middle_out = gcn_middle_out
        self.epoch_gcn = epoch_gcn
        self.k = k_neighbor
        self.mask_rate = mask_rate
        self.epoch_cpm_train = epoch_cpm_train
        self.epoch_cpm_test = epoch_cpm_test
        self.batch_size_cpm = batch_size_cpm
        self.lamb = lamb
        # self.val_size = val_size
        self.view_num = view_num
        self.view_dim = gcn_middle_out
        self.gcn_input_dim = gcn_input_dim
        self.patience = patience
        self.patience_for_query = patience_for_query
        self.early_stop = early_stop
        # 两个参数用于保存一次运行时候的embedding
        self.ref_h = None
        self.query_h = None
        self.ref_labels = None  # 这里要记录下ref data进来之后的labels（因为分成train和val的时候打乱了）
        self.gcn_models = []
        self.model_path = os.path.join(save_path, 'model')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if self.gcn_exist:
            for i in range(self.view_num):
                self.gcn_models.append(torch.load(os.path.join(self.model_path, 'gcn_model_' + str(i) + '.pt')))
        else:
            for i in range(self.view_num):
                self.gcn_models.append(scGNN(self.gcn_input_dim, self.gcn_middle_out).to(device))
        if self.cpm_exist:
            self.cpm_model = torch.load(os.path.join(self.model_path, 'cpm_model.pt'))
        else:
            self.cpm_model = CPMNets(self.view_num, self.view_dim, self.lsd, self.class_num, self.model_path,
                                     self.lamb).to(device)

    def fit(self, data, sm_arr, labels):
        '''
        :param data: 表达矩阵 (处理之后的，比如标准化), ndarray
        :param sm_arr: similarity matrix
        :param labels: 样本的标签, ndarray
        :return:
        '''

        ref_views = []
        masked_prob = min(len(data.nonzero()[0]) / (data.shape[0] * data.shape[1]), self.mask_rate)
        print(masked_prob)

        masked_data, index_pair, masking_idx = mask_data(data, masked_prob)
        graphs = [construct_graph(masked_data, sm_arr[i], self.k) for i in range(len(sm_arr))]

        if not self.gcn_exist:
            # 如果gcn_model不存在，进行gcn训练
            for i in range(self.view_num):
                self.gcn_models[i].train()
                optimizer = torch.optim.Adam(self.gcn_models[i].parameters(), lr=1e-4)
                loss_fct = nn.MSELoss()

                for epoch in range(self.epoch_gcn):
                    optimizer.zero_grad()
                    pred = self.gcn_models[i](graphs[i].to(device))

                    # 得到预测的drop out
                    dropout_pred = pred[index_pair[0][masking_idx], index_pair[1][masking_idx]]
                    dropout_true = data[index_pair[0][masking_idx], index_pair[1][masking_idx]]
                    loss = loss_fct(dropout_pred.view(-1),
                                    torch.tensor(dropout_true, dtype=torch.float).to(device).view(-1))
                    loss.backward()
                    optimizer.step()
                    if epoch % 200 == 0:
                        print('Epoch: {}, Training Loss {:.4f}'.format(epoch, loss.item()))
                    wandb.log({
                        'CGN' + "_" + str(i + 1): loss.item()
                    })
                self.gcn_models[i].eval()
                embedding = self.gcn_models[i].get_embedding(graphs[i]).detach().cpu().numpy()
                ref_views.append(z_score_scale(embedding))

            print("Save gcn models.")
            for j in range(self.view_num):
                torch.save(self.gcn_models[j], os.path.join(self.model_path, 'gcn_model_' + str(j) + '.pt'))
        else:
            # gcn 存在
            # 获得Embedding
            for i in range(self.view_num):
                # 这里相当于Batch norm?
                ref_views.append(z_score_scale(self.gcn_models[i].get_embedding(graphs[i]).detach().cpu().numpy()))

        # 在这里对ref_views做一个标准化, 并且转为tensor
        ref_data = np.concatenate(ref_views, axis=1)
        ref_data = torch.from_numpy(ref_data).float().to(device)

        # ref_views就是经过多个Similarity matrix图卷积之后的不同embeddings
        # 之后开始训练cpm net
        labels = torch.from_numpy(labels).view(-1).long().to(device)
        # 训练cpm net
        self.ref_h, self.ref_labels = self.cpm_model.train_ref_h(ref_data, labels, self.batch_size_cpm,
                                                                 self.epoch_cpm_train, self.patience, self.early_stop)

        # 这里要重新加载cpm_model, 使用早停法保留下来的泛化误差最好的模型 (后面考虑加入一个不用早停法的选项）
        if self.early_stop:
            self.cpm_model = torch.load(os.path.join(self.model_path, 'cpm_model.pt'))

        torch.save(self.cpm_model, os.path.join(self.model_path, 'cpm_model.pt'))

    def predict(self, data, sm_arr, trues):
        '''
            data: query(test)的表达矩阵（处理之后的，如mask，norm等）, ndarray
            sm_arr:
            labels: 这里labels的存在是为了帮助我们找到最好的重构epoch，实际应用场景推测的时候没有labels的话，就只能不断调整epoch

            返回预测结果（label），ndarray
        '''
        trues = torch.from_numpy(trues).view(-1).float().to(device)
        # 获得Embedding
        graphs = [construct_graph(data, sm_arr[i], self.k)
                  for i in range(self.view_num)]
        query_views = []

        for i in range(self.view_num):
            query_views.append(z_score_scale(self.gcn_models[i].get_embedding(graphs[i]).detach().cpu().numpy()))

        query_data = np.concatenate(query_views, axis=1)
        query_data = torch.from_numpy(query_data).float().to(device)

        self.query_h = self.cpm_model.train_query_h(query_data, self.epoch_cpm_test, self.early_stop, self.patience_for_query)

        pred = self.cpm_model.classify(self.query_h)
        pred_cpm = cpm_classify(self.ref_h.detach().cpu().numpy(), self.query_h.detach().cpu().numpy(),
                                self.ref_labels.detach().cpu().numpy())

        acc = (pred_cpm == trues.detach().cpu().numpy()).sum() / pred_cpm.shape[0]
        print("CPM acc is {:.3f}".format(acc))
        return pred.detach().cpu().numpy().reshape(-1)

    def get_embeddings_with_data(self, data, sm_arr, epochs):
        # 这里提供一个接口，给一个data拿到相应的embeddings
        '''
            data: list, 含有multiple-view的list
        '''
        # 获得Embedding
        graphs = [construct_graph(data, sm_arr[i], self.k)
                  for i in range(self.view_num)]
        query_views = []
        for i in range(self.view_num):
            query_views.append(torch.from_numpy(
                z_score_scale(self.gcn_models[i].get_embedding(graphs[i]).detach().cpu().numpy())))

        query_data = torch.concat(query_views, dim=1).float().to(device)
        query_h = self.cpm_model.train_query_h(query_data, epochs)
        return self.cpm_model(query_h)

    def get_ref_embeddings_and_labels(self):
        '''
            提供这个函数的目的是为了获取reconstruction中最好的epoch时的train h (这里包含了train和val）
            因为fit函数打乱了原来的labels，所以这里提供train时候的labels
        '''
        return self.cpm_model(self.ref_h), self.ref_labels

    def get_query_embeddings(self):
        '''
            这个和get_embeddings_with_data一样，只不过用起来更方便
        '''
        return self.cpm_model(self.query_h)
