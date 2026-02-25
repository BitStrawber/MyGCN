"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
import torch.nn.functional as F
from torch import nn
import numpy as np


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

class PCSRec(BasicModel):
    def __init__(self, config, dataset):
        super(PCSRec, self).__init__()
        self.config = config
        self.dataset = dataset
        self.num_users = self.dataset.m_users
        self.num_items = self.dataset.n_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        
        # 论文超参数
        self.alpha = config['alpha']
        
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
        # 路径编码可学习参数 theta (公式 1)
        self.theta = nn.Parameter(torch.ones(6) / 6.0)

        # 获取图结构
        self.Graph_pos = self.dataset.Graph_pos.to(world.device)
        self.Graph_neg = self.dataset.Graph_neg.to(world.device)
        self.X_paths = [X.to(world.device) for X in self.dataset.X_paths]
        
    def _sparse_softmax_normalize(self, indices, values, num_nodes):
        """实现公式(2) 行级稀疏 Softmax"""
        exp_values = torch.exp(values)
        row_indices = indices[0]
        row_sums = torch.zeros(num_nodes, device=values.device)
        row_sums.index_add_(0, row_indices, exp_values)
        row_sums_gathered = row_sums[row_indices]
        softmax_values = exp_values / (row_sums_gathered + 1e-8)
        return torch.sparse_coo_tensor(indices, softmax_values, (num_nodes, num_nodes))

    def computer(self):
        """
        前向传播计算最终 Embeddings
        包含 PEE 和 PCF 模块
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        num_nodes = all_emb.shape[0]

        # =======================================================
        # 1. Path-Enhanced Embedding (PEE) - 公式 (1) (2) (3)
        # =======================================================
        P_indices = None
        P_values = None
        
        # 动态计算加权路径矩阵 P
        for p in range(6):
            indices = self.X_paths[p]._indices()
            values = torch.ones_like(indices[0], dtype=torch.float32) * self.theta[p]
            
            if P_indices is None:
                P_indices = indices
                P_values = values
            else:
                # 拼接并合并重复索引
                P_indices = torch.cat([P_indices, indices], dim=1)
                P_values = torch.cat([P_values, values])
                
        # 合并重复索引的权重 (scatter add)
        P_sparse = torch.sparse_coo_tensor(P_indices, P_values, (num_nodes, num_nodes)).coalesce()
        
        # 公式(2) 逐行 Softmax 归一化
        P_matrix = self._sparse_softmax_normalize(P_sparse._indices(), P_sparse._values(), num_nodes)
        
        # 公式(3) 结构嵌入增强 E(0) <- P * E(0)
        E_0 = torch.sparse.mm(P_matrix, all_emb)
        embs = [E_0]

        # =======================================================
        # 2. Paired Channel Filtering (PCF) - 公式 (4) (5) (6)
        # =======================================================
        E_l = E_0
        for layer in range(self.n_layers):
            # 低通滤波 (正反馈通道) 公式 (4)
            E_pos = torch.sparse.mm(self.Graph_pos, E_l)
            
            # 高通滤波 (负反馈通道) 公式 (5)
            # L- = D - A- => 归一化后 E_neg = E_l - D^{-1/2} A- D^{-1/2} E_l
            E_neg_A = torch.sparse.mm(self.Graph_neg, E_l)
            E_neg = E_l - E_neg_A 
            
            # 反馈融合通道 公式 (6)
            E_l = E_pos + self.alpha * E_neg
            embs.append(E_l)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1) # 均值聚合
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.config['sigmoid'](torch.matmul(users_emb, items_emb.t()))
        return rating

    def calculate_loss(self, users, pos_items, neg_items, unobs_pos, unobs_neg):
        all_users, all_items = self.computer()

        user_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        unobs_pos_emb = all_items[unobs_pos]
        unobs_neg_emb = all_items[unobs_neg]

        # 1. 双向 BPR Loss (公式 11)
        # 预测打分
        score_pos = torch.sum(user_emb * pos_emb, dim=1)
        score_unobs_pos = torch.sum(user_emb * unobs_pos_emb, dim=1)
        score_neg = torch.sum(user_emb * neg_emb, dim=1)
        score_unobs_neg = torch.sum(user_emb * unobs_neg_emb, dim=1)

        # 正反馈：拉近喜欢的物品，推开未观测物品
        bpr_pos = -torch.log(torch.sigmoid(score_pos - score_unobs_pos) + 1e-8).mean()

        # 负反馈：推开讨厌的物品，确保未观测物品打分更高
        bpr_neg = -torch.log(torch.sigmoid(self.beta * (score_unobs_neg - score_neg)) + 1e-8).mean()

        loss_bpr = bpr_pos + bpr_neg

        # 2. Contrastive Loss (公式 7, 8, 9)
        def sim(e1, e2):
            return F.cosine_similarity(e1, e2, dim=-1) / self.tau

        P_u = torch.exp(sim(user_emb, pos_emb))
        N_u = torch.exp(sim(user_emb, neg_emb))
        loss_contra = -torch.log(P_u / (P_u + N_u + 1e-8)).mean()

        # 3. L2 正则化 (防止过拟合)
        reg_loss = (1 / 2) * (self.embedding_user.weight[users].norm(2).pow(2) +
                              self.embedding_item.weight[pos_items].norm(2).pow(2) +
                              self.embedding_item.weight[neg_items].norm(2).pow(2)) / float(len(users))

        total_loss = loss_bpr + self.gamma * loss_contra + self.config['decay'] * reg_loss
        return total_loss