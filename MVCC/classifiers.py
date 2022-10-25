import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from MVCC.util import construct_graph_with_knn
import numpy as np
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluate(val_dataloader, model):
    model.eval()
    with torch.no_grad():
        # return acc
        pred_list = []
        true_list = []
        for b_data, b_label in val_dataloader:
            logits = model(b_data)
            pred = logits.argmax(dim=1)
            pred_list.append(pred)
            true_list.append(b_label)
        preds = torch.concat(pred_list)
        trues = torch.concat(true_list)

        acc = (preds == trues).sum().item() / preds.shape[0]
        return acc

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

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

    def train_classifier(self, data, labels, patience, save_path, test_size, batch_size=128, epochs=300):
        print("Train  classifier")

        optimizer_for_classifier = optim.Adam(params=self.parameters())

        # 确定各类别的比例，用 (1-x) / (1-x).sum() 归一
        alpha = np.unique(labels, return_counts=True)[1]
        alpha = alpha / alpha.sum()
        alpha = (1 - alpha) / (1 - alpha).sum()
        criterion = nn.CrossEntropyLoss()
        # criterion = FocalLoss(alpha=alpha, gamma=1)
        train_x, val_x, train_y, val_y = train_test_split(data,
                                                          labels,
                                                          shuffle=True,
                                                          stratify=labels,
                                                          test_size=test_size,
                                                          random_state=32
                                                          )

        train_x, val_x = torch.from_numpy(train_x).float().to(device), torch.from_numpy(val_x).float().to(device)
        train_y, val_y = torch.from_numpy(train_y).long().to(device), torch.from_numpy(val_y).long().to(device)

        dataset_train = TensorDataset(train_x, train_y)
        dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
        dataset_val = TensorDataset(val_x, val_y)
        dataloader_val = DataLoader(dataset_val, shuffle=False, batch_size=batch_size)

        # 更新classifier
        val_max_acc = 0
        stop = 0
        for epoch in range(epochs):
            self.train()
            c_loss_total = 0
            train_pred_arr = []
            train_true_label = []
            for batch in dataloader_train:
                b_h, b_label = batch
                b_h = b_h.to(device)
                b_label = b_label.to(device)
                logits = self(b_h)
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
            val_acc = evaluate(dataloader_val, self)
            # if early_stop:
            if val_max_acc < val_acc:
                val_max_acc = val_acc
                stop = 0
                print(
                    'epoch {:}: train classification loss = {:.3f}, train acc is {:.3f}, val max acc is {:.3f}, save the model.'.format(
                        epoch, c_loss, train_acc, val_max_acc))

                torch.save(self, os.path.join(save_path, 'classifier.pt'))
            else:
                stop += 1
                if stop > patience:
                    print("CPM train h stop at train epoch {:}, train acc is {:.3f}, val max acc is {:.3f}".format(
                        epoch, train_acc, val_max_acc))
                    break

class CNNClassifier(Classifier):
    def __init__(self, input_dim, class_num):
        super(CNNClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.MaxPool1d(2, 2),
            nn.Conv1d(4, 8, 2, 1),
            nn.ReLU(),
            nn.Conv1d(8, 16, 2, 1),
            # nn.ReLU(),
            # nn.Conv1d(16, 32, 3, 1),
            nn.Flatten()
        )
        middle_out = 8128

        self.fcn = nn.Sequential(
            nn.Linear(middle_out, 128),
            nn.ReLU(),
            nn.Linear(128, class_num)
            # nn.Linear(256, class_num),
        )

    def forward(self, data):
        data = data.view(data.shape[0], 1, -1)
        x = self.conv(data)
        x = self.fcn(x)
        return x


class FCClassifier(Classifier):
    def __init__(self, input_dim, class_num):
        super(FCClassifier, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # nn.Linear(256, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(64, 32),
            nn.Linear(256, class_num)
        )

    def forward(self, data):
        x = self.fcn(data)
        return x


class GCNClassifier(Classifier):
    def __init__(self, input_dim, output_dim):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, 1024)
        self.conv2 = GCNConv(1024, output_dim)

    def forward(self, data):
        g_data = construct_graph_with_knn(data.detach().cpu().numpy())
        x, edge_index = g_data.x.to(device), g_data.edge_index.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

