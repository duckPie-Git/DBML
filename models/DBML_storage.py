import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
product search model
'''
class PSM_st(nn.Module):
    def __init__(self, user_size, item_size, word_size, query_size, embedding_dim,\
                 max_query_len, max_review_len, batch_size, \
                 neg_num=5,sample_num=1,transfer_hidden_dim=100,\
                 sigma_parameter=1e-7, kl_parameter=1e-7, word_parameter=1e-2, device=torch.device('cpu')):
        super(PSM, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.word_size = word_size
        self.query_size = query_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.max_query_len = max_query_len
        self.max_review_len = max_review_len
        self.sample_num = sample_num
        self.transfer_hidden_dim = transfer_hidden_dim
        self.kl_parameter = kl_parameter
        self.sigma_parameter = sigma_parameter
        self.word_parameter = word_parameter
        self.device = device
        self.neg_num = neg_num
        self.esp = 1e-7
        
        
        self.user_mean = nn.Embedding(self.user_size, self.embedding_dim)
        self.user_std = nn.Embedding(self.user_size, self.embedding_dim)
        
        self.item_mean = nn.Embedding(self.item_size, self.embedding_dim)
        self.item_std = nn.Embedding(self.item_size, self.embedding_dim)
        
        
        self.wordEmbedding_mean = nn.Embedding(self.word_size, self.embedding_dim)
        self.wordEmbedding_std = nn.Embedding(self.word_size, self.embedding_dim)
        self.queryLinear = nn.Linear(self.embedding_dim, self.embedding_dim)
    
    
        self.transfer_linear_u = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
        self.transfer_linear_i = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
        self.transfer_linear_ni = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
        self.transfer_linear_w = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
        self.transfer_linear_nw = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
#         self.transfer_linear = {
#             "u":nn.Linear(self.embedding_dim, self.transfer_hidden_dim),
#             "i":nn.Linear(self.embedding_dim, self.transfer_hidden_dim),
#             "ni":nn.Linear(self.embedding_dim, self.transfer_hidden_dim),
#             "w":nn.Linear(self.embedding_dim, self.transfer_hidden_dim),
#             'nw':nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
#         }
        self.transfer_mean_u = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_mean_i = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_mean_ni = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_mean_w = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_mean_nw = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
#         self.transfer_mean = {
#             "u":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             "i":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             "ni":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             'w':nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             'nw':nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
#         }
        self.transfer_std_u = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_std_i = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_std_ni = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_std_w = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_std_nw = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
#         self.transfer_std = {
#             "u":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             "i":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             "ni":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             'w':nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             'nw':nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
#         }
        

    
#         self.userDecoder = nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.itemDecoder = nn.Linear(self.embedding_dim, self.embedding_dim)
    '''
    (uid, pid_pos, qids_pos, len_pos, time_bin_pos)
    [( uid, pid, qids_neg, len_neg, time_bin_pos),..,( uid, pid, qids_neg, len_neg, time_bin_pos)]*neg_sample_num
    '''
    
    def forward(self, user, query, query_len, item_pos, items_neg,\
                prior_user_mean, prior_user_std,\
                prior_product_mean, prior_product_std,prior_product_neg_mean, prior_product_neg_std,\
                word, word_neg, word_len,\
                prior_word_mean, prior_word_std,\
                prior_word_neg_mean, prior_word_neg_std, loss_type='ml'):

        '''
        user
        '''
        user_mean = self.user_mean(user).squeeze(1) # (batch, out_size)
        user_std = self.user_std(user).squeeze(1).mul(0.5).exp() # (batch,ou t_size)
        

        
        '''
        query
        '''
        query = self.get_train_query_tanh_mean(query, query_len)# ((batch, maxQueryLen))) ---> ((batch, embedding)
       
        
        '''
        word
        '''
        word_mean_pos = self.wordEmbedding_mean(word)
        word_std_pos = self.wordEmbedding_std(word).mul(0.5).exp()

        '''
        neg word
        '''
        word_mean_neg = self.wordEmbedding_mean(word_neg)
        word_std_neg = self.wordEmbedding_std(word_neg).mul(0.5).exp()
    
        '''
        pos product
        '''
        item_mean_pos = self.item_mean(item_pos).squeeze(1) # (batch, out_size)
        item_std_pos = self.item_std(item_pos).squeeze(1).mul(0.5).exp() # (batch, out_size)

        
        '''
        neg product
        '''
        items_mean_neg = self.item_mean(items_neg)# (batch, neg_sample_num, out_size)
        items_std_neg = self.item_std(items_neg).mul(0.5).exp() # (batch, neg_sample_num,out_size)
        
        
        
        
        '''
        用户和product word的隐变量采样
        '''
        user_sample = self.reparameter(user_mean, user_std)
        product_sample = self.reparameter(item_mean_pos, item_std_pos)
        product_sample_neg = self.reparameter(items_mean_neg, items_std_neg)
        word_sample = self.reparameter(word_mean_pos, word_std_pos)
        word_sample_neg = self.reparameter(word_mean_neg, word_std_neg)
        
#         query_sample
        '''
        loss 计算
        '''
        # 主要的损失u+q-i 采样得到的uqi 计算重构误差
        if (loss_type == 'ml'):
            loss_main, dis_pos, dis_neg = self.lossF_sigmod_ml(user_sample, query, product_sample, product_sample_neg)
        elif (loss_type == 'sk'):
            loss_main, dis_pos, dis_neg = self.lossF_sk(user_sample, query, product_sample, product_sample_neg)
        else:
            loss_main, dis_pos, dis_neg = self.lossF_uqi(user_sample, query, product_sample, product_sample_neg)
        # 计算uw和iw的损失
        user_word_loss = self.word_loss(user_sample, word_sample, word_len, word_sample_neg)
        item_word_loss = self.word_loss(product_sample, word_sample, word_len, word_sample_neg)

        
        # 转移损失(KL损失) -->
        # 转移概率 loss current_mean, current_std, prior_mean, prior_std
        user_trans_loss = self.transfer_kl_loss(user_mean, user_std, prior_user_mean, prior_user_std, False, 'u')
        product_trans_pos_loss = self.transfer_kl_loss(item_mean_pos, item_std_pos, prior_product_mean, prior_product_std, False, 'i')
        product_trans_neg_loss = self.transfer_kl_loss(items_mean_neg, items_std_neg, prior_product_neg_mean, prior_product_neg_std, True, 'ni')
        word_trans_pos_loss = self.transfer_kl_loss(word_mean_pos, word_std_pos, prior_word_mean, prior_word_std, True, 'w')
        word_trans_pos_neg_loss = self.transfer_kl_loss(word_mean_neg, word_std_neg, prior_word_neg_mean, prior_word_neg_std, True, 'nw')
#         word_trans_pos_loss = self.transfer_kl_loss(word_mean_pos, word_std_pos, prior_word_mean, prior_word_std, False, 'w')
#         word_trans_neg_loss = self.transfer_kl_loss(word_mean_neg, word_std_neg, prior_word_neg_mean, prior_word_neg_std, True, 'nw')
        
        
        
        #         query_trans_loss
        loss = loss_main+\
        (user_word_loss+item_word_loss)*torch.tensor(self.word_parameter).to(self.device)+\
        (user_trans_loss+product_trans_pos_loss+product_trans_neg_loss+word_trans_pos_loss+word_trans_pos_neg_loss)*\
        torch.tensor(self.kl_parameter).to(self.device)
        return user_mean, user_std, query, \
                item_mean_pos, item_std_pos,\
                items_mean_neg, items_std_neg, \
                user_sample, product_sample, product_sample_neg, \
                loss, dis_pos, dis_neg,\
                word_mean_pos, word_std_pos, word_mean_neg, word_std_neg
    
    def word_loss(self, itemOrUser1, word_pos, word_len, word_neg):
        len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.max_review_len-int(i.item())) for i in word_len]).unsqueeze(2).to(self.device)
        word_pos = word_pos.mul(len_mask)
        itemOrUser = itemOrUser1.unsqueeze(1)
        dis_pos = (itemOrUser - word_pos).norm(2, dim=2).mean(dim=1)
        dis_neg = (itemOrUser - word_neg).norm(2, dim=2).mean(dim=1)
        wl = torch.log(torch.sigmoid(dis_neg-dis_pos)).mean()*(-1.0)
        return wl
        
    def reparameter(self, mean, std):
#         sigma = torch.exp(torch.mul(0.5,log_var))
        std_z = torch.from_numpy(np.random.normal(0, 1, size=std.size())).float()


        return mean + torch.tensor(self.sigma_parameter).to(self.device)*std* Variable(std_z)  # Reparameterization trick
    
    
    def get_train_query_tanh_mean(self, query, query_len):
        '''
        input size: (batch, maxQueryLen)
        对query处理使用函数
        tanh(W*(mean(Q))+b)
        
        '''
        query = self.wordEmbedding_mean(query) # size: ((batch, maxQueryLen))) ---> (batch, len(query[i]), embedding)
        # query len mask 使得padding的向量为0
        len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.max_query_len-int(i.item())) for i in query_len]).unsqueeze(2).to(self.device)
        query = query.mul(len_mask)
        query = query.sum(dim=1).div(query_len)
        query = self.queryLinear(query).tanh()
    
        return query

    def transfer_mlp(self, prior, aim='u'):
        transfer_linear = getattr(self, 'transfer_linear_'+aim)
        current_hidden = transfer_linear(prior)
        transfer_mean = getattr(self, 'transfer_mean_'+aim)
        transfer_std = getattr(self, 'transfer_std_'+aim)
        return transfer_mean(current_hidden), transfer_std(current_hidden).mul(0.5).exp()

    
    def transfer_kl_loss(self, current_mean, current_std, prior_mean, prior_std, dim3=False, aim='u'):
        dim2 = current_mean.shape[1]
        if (dim3 == False):
            current_transfer_mean = torch.zeros((self.batch_size, self.embedding_dim), device=self.device)
            current_transfer_std = torch.zeros((self.batch_size, self.embedding_dim), device=self.device)
            for i in range(self.sample_num):
                prior_instance = self.reparameter(prior_mean, prior_std)
                cur_instance = self.transfer_mlp(prior_instance, aim)
                current_transfer_mean += cur_instance[0]
                current_transfer_std += cur_instance[1]

            # 取多个采样的Q(Zt-1)分布的均值为最终的loss 计算使用的P(Zt|B1:t-1)分布
            current_transfer_mean = current_transfer_mean.div(self.sample_num)
            current_transfer_std = current_transfer_std.div(self.sample_num**2)

            kl_loss = self.DKL(current_mean, current_std, current_transfer_mean, current_transfer_std)
        else:
            current_transfer_mean = torch.zeros((self.batch_size, dim2, self.embedding_dim), device=self.device)
            current_transfer_std = torch.zeros((self.batch_size, dim2, self.embedding_dim), device=self.device)
            for i in range(self.sample_num):
                prior_instance = self.reparameter(prior_mean, prior_std)
                cur_instance = self.transfer_mlp(prior_instance, aim)
                current_transfer_mean += cur_instance[0]
                current_transfer_std += cur_instance[1]

            # 取多个采样的Q(Zt-1)分布的均值为最终的loss 计算使用的P(Zt|B1:t-1)分布
            current_transfer_mean = current_transfer_mean.div(self.sample_num)
            current_transfer_std = current_transfer_std.div(self.sample_num)

            kl_loss = self.DKL(current_mean, current_std, current_transfer_mean, current_transfer_std, True)
        
        return kl_loss
    
    
    '''
    KL 误差
    KL(Q(Zt)||P(Zt|B1:t-1))
    P(Zt|B1:t-1) 使用采样计算～～1/K sum_{i=1}^K(P(Zt|Z_{i}t-1))
    '''
    def DKL(self, mean1, std1, mean2, std2, neg = False):
        var1 = std1.pow(2)
        var2 = std2.pow(2)
        mean_pow2 = (mean2-mean1)*(torch.tensor(1.0)/var2)*(mean2-mean1)
        tr_std_mul = (torch.tensor(1.0)/var2)*var1
        if (neg == False):
            dkl = (torch.log(var2/var1 + self.esp)-1+tr_std_mul+mean_pow2).mul(0.5).sum(dim=1).mean()
        else:
            dkl = (torch.log(var2/var1 + self.esp)-1+tr_std_mul+mean_pow2).mul(0.5).sum(dim=2).sum(dim=1).mean()
        return dkl
    
    '''
    主损失 重构误差
    -Eq(log{P(Bt|Zt)})
    '''
    def lossF_sk(self, user, query, item_pos, items_neg):    
        u_plus_q = user+query
        uq=u_plus_q.unsqueeze(2)  # skip
        itp = item_pos.unsqueeze(1) # skip
        pos_skip = torch.bmm(itp, uq) # skip
        loss_pos = pos_skip.sigmoid().log().mean()
        itn = items_neg.unsqueeze(2) # skip
        batch_size, neg_num, em_dim = items_neg.shape
        neg_skip = torch.empty(batch_size, neg_num, 1)
        for i in range(self.batch_size):
            neg_skip[i] = torch.matmul(itn[i],uq[i]).squeeze(2)
        loss_neg = neg_skip.mul(-1.0).sigmoid().log().sum(dim=1).mean()
        batch_loss = -1.0*(loss_pos + loss_neg)
        return batch_loss, pos_skip.mean(), neg_skip.mean()
    def lossF_sigmod_ml(self, user, query, item_pos, items_neg):
        u_plus_q = user+query
        dis_pos = (u_plus_q - item_pos).norm(2, dim=1).mul(5.)
        u_plus_q.unsqueeze_(1)
        dis_neg = (u_plus_q - items_neg)
        dis_neg = dis_neg.norm(2,dim=2)
        dis_pos = dis_pos.view(-1,1)
        batch_loss = torch.log(torch.sigmoid(dis_neg-dis_pos)).sum(dim=1)*(-1.0)
        return batch_loss.mean() , dis_pos.mean(), dis_neg.mean()
    def lossF_uqi(self, user, query, item_pos, items_neg):
        u_plus_q = user+query
        dis_pos = (u_plus_q - item_pos).norm(2, dim=1).mul(-1)
        log_u_plus_q_minus_i_pos = dis_pos.sigmoid().log()
        u_plus_q.unsqueeze_(1)
        dis_neg = (u_plus_q - items_neg)
        dis_neg = dis_neg.norm(2,dim=2)
        log_u_plus_q_minus_i_neg = dis_neg.sigmoid().log().mean(dim=1)
        batch_loss = -1*(0.1*log_u_plus_q_minus_i_neg + log_u_plus_q_minus_i_pos)
        return batch_loss.mean() , dis_pos.mean(), dis_neg.mean()