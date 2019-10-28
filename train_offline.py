#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pickle
from torch.utils.data import DataLoader, Dataset


# In[ ]:


import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(3)


# In[ ]:


from file_utils.mydata import MyData


# In[ ]:


data_name = 'Electronics'


# In[ ]:


with open('input_data/dataset_time_'+data_name+'.bin', 'rb') as f:
    data_set = pickle.load(f)


# In[ ]:


'''
dataset statistic
'''
data_set.productNum, data_set.userNum, data_set.wordNum, len(data_set.train_data)


# In[ ]:


from models.DBML_offline import PSM


# ## 定义参数 

# In[ ]:


'''
实验参数
'''
embedding_dim = 50
out_size = 10
batch_size = 256
neg_sample_num = data_set.neg_sample_num
dataLen = len(data_set.train_data)
batch_num = int(dataLen/batch_size)
full_len = batch_num*batch_size
time_bin_num = len(data_set.time_data)
total_epoch = 2
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device  = torch.device("cpu")


# ## 定义模型

# In[ ]:


dbml = PSM(data_set.userNum,
           data_set.productNum,
           data_set.wordNum,
           embedding_dim,
           data_set.max_query_len,
           data_set.max_review_len,
           batch_size,
           data_set.time_num + 1,
           neg_num=5,
           sample_num=1,
           transfer_hidden_dim=100,
           sigma_parameter=1e-3,
           kl_parameter=1e-3,
           word_parameter=1e0,
           device=device)
dbml.to(device)


# In[ ]:


# dbml = torch.nn.DataParallel(dbml)


# ### load model

# In[ ]:


# dbml.load_state_dict(torch.load('./out/Electronics_2019-07-22-13-05-52_success.pkl'))


# ### 加载数据

# In[ ]:


device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dbml.to(device)
data_gen = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=True)


# ### 定义优化器

# In[ ]:


device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dbml.to(device)
optimizer = torch.optim.Adam(dbml.parameters(), lr=0.00001)


# In[ ]:


dbml.kl_parameter


# In[ ]:


dbml.train()
total_epoch = 500
total_batch = len(data_gen)
for e in range(total_epoch):
    for i, data in enumerate(data_gen):
        
        user_mean, user_std, query, item_mean_pos, item_std_pos, items_mean_neg, items_std_neg,        user_sample, product_sample, product_sample_neg, loss, dis_pos, dis_neg, word_mean_pos,        word_std_pos, word_mean_neg, word_std_neg         = dbml(
        data[0][0].to(device), data[0][1].to(device),
        torch.stack(data[0][2]).t().to(device), data[0][3].to(device),
        torch.stack(data[0][4]).t().to(device), data[0][5].to(device), data[0][6].to(device),
        torch.stack(data[1][0]).t().to(device),
        torch.stack(data[1][1]).t().to(device))
        
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()
        if (i % 20 == 0):
            print('E: {}/{} | B: {}/{} | Loss: {} | POS: {} | NEG: {}'.format(e, total_epoch, i, total_batch,loss[0].item(), dis_pos.item(), dis_neg.item()))
            print('Loss:{} | Main:{} | Word:{} | KL:{}'.format(loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item()))


# ### 模型保存

# In[ ]:


e


# In[ ]:


import time as tt


# In[ ]:


torch.save(dbml.state_dict(), 'out/{}_{}_{}.pkl'.format( data_name, tt.strftime("%Y-%m-%d-%H-%M-%S", tt.localtime()),'ok'))


# In[ ]:





# ### 测试模型

# In[ ]:


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.
def mean_reciprocal_rank(r):
    return np.sum(r / np.arange(1, r.size + 1))
def hit_rate(r):
    if (np.sum(r) >= 0.9):
        return 1
    else:
        return 0


# In[ ]:


def get_query_laten(q_linear, query, query_len, max_query_len):
    '''
    input size: (batch, maxQueryLen)
    对query处理使用函数
    tanh(W*(mean(Q))+b)
    '''
    query_len = torch.tensor(query_len).view(1,-1).float()
    # size: ((batch, maxQueryLen))) ---> (batch, len(query[i]), embedding)
    # query len mask 使得padding的向量为0
    len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(max_query_len-int(i.item())) for i in query_len]).unsqueeze(2)
    query = query.mul(len_mask)
    query = query.sum(dim=1).div(query_len)
    query = q_linear(query).tanh()

    return query


# In[ ]:


dbml.eval()
device=torch.device('cpu')
dbml.to(device)
all_p_m = torch.empty(data_set.time_num, data_set.productNum, embedding_dim)
for ii in range(data_set.time_num):
    for i in range(data_set.productNum):
        p_mean = dbml.item_mean(torch.tensor([i], device=device)).squeeze(1)
        time= dbml.time_embdding(torch.tensor([ii], device=device)+torch.tensor(1, device=device)).squeeze(1)
        p_mean = dbml.time2mean(torch.cat([p_mean, time], 1)).squeeze()
        all_p_m[ii][i] = p_mean


# In[ ]:


from tqdm import trange


# In[ ]:


eval_dataset = data_set.test_data
test_counter = 0
all_hr = 0
all_ndcg = 0
all_mrr = 0
for ii in trange(len(eval_dataset)):
    td = eval_dataset[ii]
    '''
    应该定义一个训练过的user， 这里简单的先取训练过的时间段的用户
    '''
    if (td[6] >= 0):
        user = dbml.user_mean(torch.tensor([td[0]], device=device)).squeeze(1)
        time= dbml.time_embdding(torch.tensor([td[6]], device=device)+torch.tensor(1, device=device)).squeeze(1)
        user = dbml.time2mean(torch.cat([user, time], 1)).squeeze()
        
        query_len = td[3]
        query = torch.cat(tuple([dbml.wordEmbedding_mean(torch.tensor([i], device=device).squeeze(0)) for i in td[2]])).view(1,-1,embedding_dim)
        query = get_query_laten(dbml.queryLinear, query, query_len, data_set.max_query_len)
        user_query = user+query
#         uq_i = torch.empty(datasets.productNum)
        user_query.squeeze_(0)
        uq_i = (user_query - all_p_m[td[6]]).norm(2, dim=1)*(-1.)
#         for i in range(datasets.productNum):
#             p_mean = product_time_latent[td[6]+1][i][0]
#             uq_i[i] = -1*(user_query-p_mean).norm(2).item()
        ranks_order = uq_i.topk(20)[1]
        r = torch.eq(ranks_order, td[1]).numpy()
        all_hr += hit_rate(r)
        all_mrr += mean_reciprocal_rank(r)
        all_ndcg += dcg_at_k(r, 20, 1)
        test_counter += 1
hr = all_hr / float(test_counter+1e-6)
mrr = all_mrr / float(test_counter+1e-6)
ndcg = all_ndcg / float(test_counter+1e-6)
print(hr, mrr, ndcg)


# In[ ]:


len(eval_dataset)


# In[ ]:


data_set.eval_data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




