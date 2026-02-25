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
        self.dataset: BasicDataset = dataset
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
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.decay = config['decay']
        self.lr = config['lr']
        
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.theta = nn.Parameter(torch.ones(6) / 6.0)
        self.f = nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.Graph_pos, self.Graph_neg, self.PathMats = self.dataset.getSignedGraphComponents()
        self.Graph_pos = self.Graph_pos.coalesce().to(world.device)
        self.Graph_neg = self.Graph_neg.coalesce().to(world.device)
        self.PathMats = [mat.coalesce().to(world.device) for mat in self.PathMats]
        
    def _sparse_softmax_normalize(self, indices, values, num_nodes):
        """实现公式(2) 行级稀疏 Softmax"""
        exp_values = torch.exp(values)
        row_indices = indices[0]
        row_sums = torch.zeros(num_nodes, device=values.device)
        row_sums.index_add_(0, row_indices, exp_values)
        row_sums_gathered = row_sums[row_indices]
        softmax_values = exp_values / (row_sums_gathered + 1e-12)
        return torch.sparse_coo_tensor(indices, softmax_values, (num_nodes, num_nodes)).coalesce()

    def _build_path_weighted_matrix(self, num_nodes):
        theta_weight = torch.softmax(self.theta, dim=0)
        all_indices = []
        all_values = []
        for path_id, path_mat in enumerate(self.PathMats):
            path_mat = path_mat.coalesce()
            indices = path_mat.indices()
            if indices.shape[1] == 0:
                continue
            values = torch.full(
                (indices.shape[1],),
                theta_weight[path_id],
                device=indices.device,
                dtype=torch.float32
            )
            all_indices.append(indices)
            all_values.append(values)
        if len(all_indices) == 0:
            eye_idx = torch.arange(num_nodes, device=world.device, dtype=torch.long)
            indices = torch.stack([eye_idx, eye_idx], dim=0)
            values = torch.ones(num_nodes, device=world.device, dtype=torch.float32)
            return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

        cat_indices = torch.cat(all_indices, dim=1)
        cat_values = torch.cat(all_values, dim=0)
        weighted = torch.sparse_coo_tensor(cat_indices, cat_values, (num_nodes, num_nodes)).coalesce()
        return self._sparse_softmax_normalize(weighted.indices(), weighted.values(), num_nodes)

    def computer(self):
        """
        前向传播计算最终 Embeddings
        包含 PEE 和 PCF 模块
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        num_nodes = all_emb.shape[0]

        P_matrix = self._build_path_weighted_matrix(num_nodes)
        E_0 = torch.sparse.mm(P_matrix, all_emb)
        embs = [E_0]

        E_l = E_0
        for _ in range(self.n_layers):
            E_pos = torch.sparse.mm(self.Graph_pos, E_l)
            E_neg_A = torch.sparse.mm(self.Graph_neg, E_l)
            E_neg = E_l - E_neg_A 
            E_l = E_pos + self.alpha * E_neg
            embs.append(E_l)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def calculate_loss(self, users, pos_items, neg_items, unobs_pos, unobs_neg):
        all_users, all_items = self.computer()
        
        user_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        unobs_pos_emb = all_items[unobs_pos]
        unobs_neg_emb = all_items[unobs_neg]

        pos_scores = torch.mul(user_emb, pos_emb).sum(dim=1)
        unobs_pos_scores = torch.mul(user_emb, unobs_pos_emb).sum(dim=1)
        bpr_pos = F.softplus(unobs_pos_scores - pos_scores).mean()

        neg_scores = torch.mul(user_emb, neg_emb).sum(dim=1)
        unobs_neg_scores = torch.mul(user_emb, unobs_neg_emb).sum(dim=1)
        bpr_neg = F.softplus(self.beta * (neg_scores - unobs_neg_scores)).mean()

        loss_bpr = bpr_pos + bpr_neg

        def sim(e1, e2):
            return F.cosine_similarity(e1, e2, dim=-1) / self.tau

        P_u = torch.exp(sim(user_emb, pos_emb))
        N_u = torch.exp(sim(user_emb, neg_emb))
        
        loss_contra = -torch.log(P_u / (P_u + N_u + 1e-12)).mean()

        user_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        reg_loss = (1/2)*(user_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2))/float(len(users))
        
        total_loss = loss_bpr + self.gamma * loss_contra + self.decay * reg_loss
        
        return total_loss

    def bpr_loss(self, users, pos, neg):
        raise NotImplementedError("PCSRec 使用 calculate_loss 与双损失训练流程")