import os.path
import copy
import numpy as np
import torch.nn as nn
import torch
import wandb
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import pdist
from MVCC.util import cpm_classify, mask_data, construct_graph, z_score_scale, construct_graph_with_self
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import silhouette_score
from torch.utils.data import Dataset, TensorDataset

'''
    CPM-Nets, 改写为Pytroch形式
'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


class CNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, class_num):
        super(CNNClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.MaxPool1d(2, 2),
            nn.Conv1d(8, 16, 3, 1),
            nn.ReLU(),
            # nn.Conv1d(16, 32, 3, 1),
            # nn.ReLU(),
            nn.Flatten()
        )
        middle_out = int((input_dim - 4) * 16)
        self.fcn = nn.Sequential(
            nn.Linear(middle_out, 2048),
            nn.ReLU(),
            nn.Linear(2048, class_num)
        )

    def forward(self, data):
        data = data.view(data.shape[0], 1, -1)
        x = self.conv(data)
        x = self.fcn(x)
        return x


class FCClassifier(torch.nn.Module):
    def __init__(self, input_dim, class_num):
        super(FCClassifier, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

    def forward(self, data):
        x = self.fcn(data)
        return x


class GCNClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, 1024)
        self.conv2 = GCNConv(1024, output_dim)

    def forward(self, data):
        g_data = construct_graph_with_self(data.detach().cpu().numpy()).to(device)
        x, edge_index = g_data.x.to(device), g_data.edge_index.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x



class CPMNets(torch.nn.Module):
    def __init__(self, view_num, view_dim, lsd, class_num, save_path):
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
        # model save path
        self.save_path = save_path
        # net的输入是representation h ---> 重构成 X
        # 重构的网络
        self.net = []
        self.ref_h = None
        self.ref_labels = None
        self.query_h = None

        # self.classifiers = []
        for i in range(view_num):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.lsd, self.view_dim, device=device),
                )
            )

        # 分类的网络
        self.class_num = class_num
        # self.classifier = FCClassifier(self.lsd, self.class_num)
        self.classifier = GCNClassifier(self.lsd, self.class_num)
        self.classifier = self.classifier.to(device)

    def reconstrution_loss(self, r_x, x):
        '''
        :param r_x: 由 h 重构出来的x
        :param x:  原x
        :return: 返回 (r_x-x)^2
        '''
        return ((r_x - x) ** 2).sum()

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

    def train_ref_h(self,
                    ref_data,
                    ref_label,
                    batch_size_classifier,
                    epochs_cpm_ref,
                    epochs_classifier,
                    gamma=1,
                    patience_for_classifier=100,
                    patience_for_cpm_ref=100,
                    test_size=0.2,
                    lamb=500,
                    ):

        '''
        这个函数直接对模型进行训练
        随着迭代，更新net，h以及classifier
        在更新h的时候，可以考虑加入原论文的classification loss
        :param ref_data: training data, ndarray, not a list
        :param ref_label:
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
        train_data = ref_data
        train_label = ref_label.view(-1)
        # 只为train data创建h, val h由后面的评估生成
        # print(labels.shape)
        train_h = torch.zeros((train_data.shape[0], self.lsd), dtype=torch.float).to(device)
        train_h.requires_grad = True  # 先放在GPU上再设置requires_grad
        nn.init.xavier_uniform_(train_h)

        optimizer_for_net = optim.Adam(params=net_params)
        optimizer_for_train_h = optim.Adam(params=[train_h])
        optimizer_for_classifier = optim.Adam(params=self.classifier.parameters())

        # 确定各类别的比例，用 (1-x) / (1-x).sum() 归一
        alpha = torch.unique(ref_label, return_counts=True)[1]
        alpha = alpha / alpha.sum()
        alpha = (1 - alpha) / (1 - alpha).sum()
        # gamma=0, 退化为带平衡因子的CE
        # 这个Focal loss在pltaform task1里面带来了很大的作用
        # criterion = FocalLoss(gamma=gamma, alpha=alpha)
        criterion = nn.CrossEntropyLoss()

        # 训练net和h
        # 设置model为train模式
        for i in range(self.view_num):
            self.net[i].train()

        min_c_loss = 99999999
        min_c_loss_train_h = None
        stop = 0

        for epoch in range(epochs_cpm_ref):
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
            # c_loss += self.fisher_loss(train_h, train_label)
            total_loss = r_loss + lamb * c_loss

            optimizer_for_train_h.zero_grad()
            total_loss.backward()
            optimizer_for_train_h.step()

            # 早停法
            if c_loss < min_c_loss:
                stop = 0
                min_c_loss = c_loss
                min_c_loss_train_h = train_h.detach().clone()
                for i in range(self.view_num):
                    torch.save(self.net[i], os.path.join(self.save_path, 'cpm_recon_net_' + str(i) + '.pt'))
                if epoch % 100 == 0:
                    print(
                        'epoch %d: Reconstruction loss = %.3f, classification loss = %.3f.' % (
                            epoch, r_loss.detach().item(), c_loss.detach().item()))
            else:
                stop += 1
                if stop > patience_for_cpm_ref:
                    print("CPM train stop at epoch {:}, min classification loss is {:.3f}".format(epoch, min_c_loss))
                    break

        # 重新加载保存好的reconstruction net以及最优的train_h
        for i in range(self.view_num):
            self.net[i] = torch.load(os.path.join(self.save_path, 'cpm_recon_net_' + str(i) + '.pt'))
        train_h = min_c_loss_train_h.detach().clone()

        '''
            classifier 训练
        '''
        train_h.requires_grad = False
        for i in range(self.view_num):
            for param in self.net[i].parameters():
                param.requires_grad = False

        print("Train  classifier")
        train_x, val_x, train_y, val_y = train_test_split(train_h.detach().cpu().numpy(),
                                                          train_label.detach().cpu().numpy(),
                                                          shuffle=True,
                                                          stratify=train_label.detach().cpu().numpy(),
                                                          test_size=test_size,
                                                          random_state=32
                                                          )

        train_x, val_x = torch.from_numpy(train_x).float().to(device), torch.from_numpy(val_x).float().to(device)
        train_y, val_y = torch.from_numpy(train_y).long().to(device), torch.from_numpy(val_y).long().to(device)

        dataset_train = TensorDataset(train_x, train_y)
        dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size_classifier)
        dataset_val = TensorDataset(val_x, val_y)
        dataloader_val = DataLoader(dataset_val, shuffle=False, batch_size=batch_size_classifier)

        # 更新classifier
        val_max_acc = 0
        stop = 0
        patience = patience_for_classifier
        for epoch in range(epochs_classifier):
            self.classifier.train()
            c_loss_total = 0
            train_pred_arr = []
            train_true_label = []
            for batch in dataloader_train:
                b_h, b_label = batch
                b_h = b_h.to(device)
                b_label = b_label.to(device)
                # CNN
                # logits = self.classifier(b_h.view(b_h.shape[0], 1, -1))
                # FCN
                # logits = self.classifier(b_h)
                # GCN
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
            val_acc = self.evaluate(dataloader_val, self.classifier)
            # if early_stop:
            if val_max_acc < val_acc:
                val_max_acc = val_acc
                stop = 0
                print(
                    'epoch {:}: train classification loss = {:.3f}, train acc is {:.3f}, val max acc is {:.3f}, save the model.'.format(
                        epoch, c_loss, train_acc, val_max_acc))

                torch.save(self.classifier, os.path.join(self.save_path, 'classifier.pt'))
            else:
                stop += 1
                if stop > patience:
                    print("CPM train h stop at train epoch {:}, train acc is {:.3f}, val max acc is {:.3f}".format(
                        epoch, train_acc, val_max_acc))
                    break

        self.ref_h = train_h
        self.ref_labels = train_label
        # 加载已保存好的classifier
        self.classifier = torch.load(os.path.join(self.save_path, 'classifier.pt'))

        return min_c_loss_train_h, train_label

    def train_query_h(self, data, n_epochs, patience_for_cpm_query):
        '''
        :param data: query data, not a list
        :param n_epochs: epochs for reconstruction
        :return:
        '''
        data = data.to(device)

        h_test = torch.zeros((data.shape[0], self.lsd), dtype=torch.float).to(device)
        h_test.requires_grad = True
        nn.init.xavier_uniform_(h_test)
        optimizer_for_query_h = optim.Adam(params=[h_test], lr=1e-2)
        for i in range(self.view_num):
            self.net[i] = self.net[i].to(device)
            self.net[i].eval()

        # 数据准备
        # dataset = MultiViewDataSet(multi_view_data, None, h_test, self.view_num)
        # dataloader = DataLoader(dataset, shuffle=False, batch_size=128)

        min_r_loss = 999999999
        min_r_loss_test_h = None
        stop = 0
        for epoch in range(n_epochs):
            r_loss = 0
            for i in range(self.view_num):
                r_loss += self.reconstrution_loss(self.net[i](h_test),
                                                  data[:, i * self.view_dim: (i + 1) * self.view_dim])
            optimizer_for_query_h.zero_grad()
            r_loss.backward()
            optimizer_for_query_h.step()

            if r_loss < min_r_loss:
                min_r_loss = r_loss
                min_r_loss_test_h = h_test.detach().clone()
                stop = 0
                if epoch % 100 == 0:
                    print('epoch {:} CPM query h: reconstruction loss {:}'.format(epoch, r_loss.detach().item()))

            else:
                stop += 1
                if stop > patience_for_cpm_query:
                    print("train query h stop at epoch {:}, min r loss {:.3f}".format(epoch, min_r_loss))
                    break

        return min_r_loss_test_h

    def evaluate(self, val_dataloader, model):
        model.eval()
        with torch.no_grad():
            # output = self.classifier(val_g_data)
            # preds = output.detach().argmax(dim=1)
            # acc = (preds==val_label).sum().item() / preds.shape[0]
            # return acc
            pred_list = []
            true_list = []
            for b_data, b_label in val_dataloader:
                # logits = model(b_data)

                logits = self.classifier(b_data)
                # logits = model(b_data.view(b_data.shape[0], 1, -1))
                pred = logits.argmax(dim=1)
                pred_list.append(pred)
                true_list.append(b_label)
            preds = torch.concat(pred_list)
            trues = torch.concat(true_list)

            acc = (preds == trues).sum().item() / preds.shape[0]
            return acc

    def classify(self, h):
        # for model in self.classifiers:
        #     model.eval()
        self.classifier = self.classifier.to(device)
        self.classifier.eval()
        # dataset = TensorDataset(h)
        # logits_arr = []
        with torch.no_grad():
            logits = self.classifier(h)
            # logits = self.classifier(h.view(h.shape[0], 1, -1))
            # logits = self.classifier(h)

            # logits = torch.concat(logits_arr, dim=0)
            # logits = logits.view(-1, h.shape[0], self.class_num)
            # logits = logits.mean(dim=0)
            pred = logits.argmax(dim=1)
            return pred

    def forward(self, data):
        # return self.classifier(data)
        # g_data = construct_graph_with_self(data.detach().cpu().numpy()).to(device)
        # return self.classifier(data.view(data.shape[0], 1, -1))
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
        x, edge_index = G_data.x.to(device), G_data.edge_index.to(device)
        # 中间夹了一层relu和一层dropout(避免过拟合的发生)
        # print(G_data.edge_index)
        x = F.relu(self.conv1(x, edge_index))
        # 可以调整dropout的比率
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_embedding(self, G_data):
        x, edge_index = G_data.x, G_data.edge_index
        return self.conv1(x.to(device), edge_index.to(device))


class MVCCModel(nn.Module):

    def __init__(self,
                 lsd,
                 class_num,
                 view_num=4,
                 save_path='',
                 label_encoder=None,
                 ):
        '''
        :param pretrained:

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
        super(MVCCModel, self).__init__()
        self.lsd = lsd
        self.class_num = class_num
        self.view_num = view_num
        # 两个参数用于保存一次运行时候的embedding
        self.ref_h = None
        self.query_h = None
        self.ref_labels = None  # 这里要记录下ref data进来之后的labels（因为分成train和val的时候打乱了）
        self.gcn_models = []
        self.cpm_model = None
        # self.min_mask_rate = min_mask_rate
        self.model_path = os.path.join(save_path, 'model')
        self.label_encoder = label_encoder

    def train_gcn(self, graph_data, model, data,
                  index_pair, masking_idx, i,
                  epoch_gcn, patience_for_gcn, save_path):
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fct = nn.MSELoss()

        stop = 0
        min_r_loss = 999999999
        best_model = None
        for epoch in range(epoch_gcn):
            optimizer.zero_grad()
            pred = model(graph_data.to(device))

            # 得到预测的drop out
            dropout_pred = pred[index_pair[0][masking_idx], index_pair[1][masking_idx]]
            dropout_true = data[index_pair[0][masking_idx], index_pair[1][masking_idx]]
            loss = loss_fct(dropout_pred.view(-1),
                            torch.tensor(dropout_true, dtype=torch.float).to(device).view(-1))
            loss.backward()
            optimizer.step()

            # 早停法
            loss_item = loss.item()
            if loss_item < min_r_loss:
                min_r_loss = loss_item
                stop = 0
                # torch.save(model.state_dict(), os.path.join(save_path,'gcn_model_'+str(i)+'_state.pt'))
                # best_model = copy.deepcopy(model)
                if epoch % 10 == 0:
                    print('View {:} Epoch: {}, Training Loss {:.4f}'.format(i, epoch, loss.item()))
            else:
                stop += 1
                if stop > patience_for_gcn:
                    print("View {:} stop at epoch {:}, min r loss {:.3f}".format(i, epoch, min_r_loss))
                    break

        # 保存模型
        torch.save(model, os.path.join(save_path, 'gcn_model_' + str(i) + '.pt'))
        # model = torch.load(os.path.join(save_path, 'gcn_model_' + str(i) + '.pt'))
        model.eval()
        embedding = model.get_embedding(graph_data).detach().cpu().numpy()
        return z_score_scale(embedding)

    def fit(self, data, sm_arr, labels,
            gcn_input_dim, gcn_middle_out,
            exp_mode=2,
            epoch_gcn=3000,
            k_neighbor=2,
            lamb=500,
            epoch_cpm_ref=3000,
            epoch_classifier=1000,
            patience_for_classifier=100,
            batch_size_classifier=256,
            mask_rate=0.3,
            gamma=1,
            test_size=0.2,
            patience_for_cpm_ref=200,
            patience_for_gcn=200,
            ):

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # 初始化内部的类
        if exp_mode == 1:
            # start from scratch
            self.cpm_model = CPMNets(self.view_num,
                                     gcn_middle_out,
                                     self.lsd,
                                     self.class_num,
                                     self.model_path).to(device)
            for i in range(self.view_num):
                self.gcn_models.append(scGNN(gcn_input_dim, gcn_middle_out).to(device))

        elif exp_mode == 2:
            # Multi reference
            pass
        elif exp_mode == 3:
            # MVCC model exist, GCN exist, cpm not (to test rest part)
            self.cpm_model = CPMNets(self.view_num,
                                     gcn_middle_out,
                                     self.lsd,
                                     self.class_num,
                                     self.model_path).to(device)
            gcn_model_paths = ['gcn_model_0.pt',
                               'gcn_model_1.pt',
                               'gcn_model_2.pt',
                               'gcn_model_3.pt']
            for i in range(self.view_num):
                self.gcn_models.append(torch.load(os.path.join(self.model_path, gcn_model_paths[i])))

        '''
            训练自监督GCN, 获取ref views
        '''
        ref_views = []
        masked_prob = min(len(data.nonzero()[0]) / (data.shape[0] * data.shape[1]), mask_rate)

        print("gcn mask prob is {:.3f}".format(masked_prob))

        masked_data, index_pair, masking_idx = mask_data(data, masked_prob)

        if exp_mode == 1:
            # start from sratch
            for i in range(self.view_num):
                graph_data = construct_graph(masked_data, sm_arr[i], k_neighbor)

                embeddings = self.train_gcn(graph_data, self.gcn_models[i], data,
                                            index_pair, masking_idx, i,
                                            epoch_gcn, patience_for_gcn, self.model_path)
                ref_views.append(embeddings)
        elif exp_mode == 2:
            # multi ref
            for i in range(self.view_num):
                graph_data = construct_graph(masked_data, sm_arr[i], k_neighbor)

                embeddings = self.train_gcn(graph_data, self.gcn_models[i], data,
                                            index_pair, masking_idx, i,
                                            epoch_gcn, patience_for_gcn, self.model_path)
                ref_views.append(embeddings)

        elif exp_mode == 3:
            # GCN exist, only to test rest part of experiment (CPM net, classifier)
            for i in range(self.view_num):
                graph_data = construct_graph(masked_data, sm_arr[i], k_neighbor)
                ref_views.append(z_score_scale(self.gcn_models[i].get_embedding(graph_data).detach().cpu().numpy()))

        ref_data = np.concatenate(ref_views, axis=1)
        ref_data = torch.from_numpy(ref_data).float().to(device)
        ref_labels = torch.from_numpy(labels).view(-1).long().to(device)
        '''
            训练CPM net
        '''

        self.ref_h, self.ref_labels = self.cpm_model.train_ref_h(ref_data,
                                                                 ref_labels,
                                                                 batch_size_classifier,
                                                                 epochs_cpm_ref=epoch_cpm_ref,
                                                                 epochs_classifier=epoch_classifier,
                                                                 gamma=gamma,
                                                                 patience_for_classifier=patience_for_classifier,
                                                                 test_size=test_size,
                                                                 lamb=lamb,
                                                                 patience_for_cpm_ref=patience_for_cpm_ref
                                                                 )

        # 这里要重新加载cpm_model, 使用早停法保留下来的泛化误差最好的模型
        # self.cpm_model = torch.load(os.path.join(self.model_path, 'cpm_model.pt'))
        #
        # torch.save(self.cpm_model, os.path.join(self.model_path, 'cpm_model.pt'))
        # 实验
        # pred = self.predict(None, None, query_views=query_views)
        # return pred

    def predict(self, data, sm_arr, epoch_cpm_query=500, k_neighbor=3, patience_for_cpm_query=100):
        '''
            data: query(test)的表达矩阵（处理之后的，如mask，norm等）, ndarray
            sm_arr:
            labels: 这里labels的存在是为了帮助我们找到最好的重构epoch，实际应用场景推测的时候没有labels的话，就只能不断调整epoch

            返回预测结果（label），ndarray
        '''
        # trues = torch.from_numpy(trues).view(-1).float().to(device)
        # 获得Embedding
        graphs = [construct_graph(data, sm_arr[i], k_neighbor)
                  for i in range(self.view_num)]
        query_views = []

        for i in range(self.view_num):
            self.gcn_models[i] = self.gcn_models[i].to(device)
            self.gcn_models[i].eval()
            with torch.no_grad():
                query_views.append(z_score_scale(self.gcn_models[i].get_embedding(graphs[i]).detach().cpu().numpy()))

        query_data = np.concatenate(query_views, axis=1)
        query_data = torch.from_numpy(query_data).float().to(device)

        self.query_h = self.cpm_model.train_query_h(query_data, epoch_cpm_query, patience_for_cpm_query)

        pred = self.cpm_model.classify(self.query_h)
        # CPM acc
        # pred_cpm = cpm_classify(self.ref_h.detach().cpu().numpy(), self.query_h.detach().cpu().numpy(),
        #                         self.ref_labels.detach().cpu().numpy())

        # acc = (pred_cpm == trues.detach().cpu().numpy()).sum() / pred_cpm.shape[0]
        # print("CPM acc is {:.3f}".format(acc))
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
