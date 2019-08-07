from torch.utils.data import DataLoader, Dataset
import numpy as np
class MyData(Dataset):
    def __init__(self, reviewData, metaData, neg_sample_num, max_query_len, max_review_len, time_num, weights = True):
        
        
        self.id2user = dict()
        self.user2id = dict()
        
        self.id2product = dict()
        self.product2id = dict()
        
        self.product2query = dict()
        
        # query
        self.word2id = dict()
        self.id2word = dict()
        
        
#         self.userText = dict()
        
        self.userReviews = dict()
        self.userReviewsCount = dict()
        self.userReviewsCounter = dict()
        self.userReviewsTest = dict()
        
        
        self.nes_weight = []
        self.word_weight = []
        self.max_review_len = max_review_len
        self.max_query_len = max_query_len
        self.neg_sample_num = neg_sample_num
        
        self.time_num = time_num
        self.time_data = []
        
        self.init_dict(reviewData, metaData)
        

        self.train_data = []
        self.test_data = []
        self.eval_data = []

        self.init_dataset(reviewData, weights)
        self.init_sample_table()
    def init_dict(self, reviewData, metaData):
        for i in range(self.time_num):
            self.time_data.append([])
        
        uid = 0
        us = set(reviewData['reviewerID'])
        pr = set()
        words = set()
        for u in us:
            # 只有两个购买记录 不够验证和测试
            asins = list(reviewData[reviewData['reviewerID'] == u]['asin'])
            if (len(asins) <= 2):
                continue

            self.id2user[uid] = u
            self.user2id[u] = uid

            # 得到每个用户购买物品记录
            pr.update(asins)
            self.userReviews[uid] = asins
            #　最后一个购买的物品做测试集
            self.userReviewsTest[uid] = asins[-1]
            words.update(set(' '.join(list(review_datas[review_datas['reviewerID'] == u]['reviewText'])).split()))
#             reviewTexts += list(reviewData[reviewData['reviewerID'] == u]['reviewText'])
            uid += 1
            if uid % 100 == 0:
                with open (r'out.txt','a+') as ff:
                    ff.write(str(len(us))+' uid: '+str(uid)+'\n')
        self.userNum = uid
        
        pid = 0
#         words = set()
        for p in pr:
            if pid % 300 == 0:
                with open (r'out.txt','a+') as ff:
                    ff.write(str(len(pr)) + ' pid:'+str(pid)+'\n')
            try:
                '''
                判断这个product是否有query
                '''
                if (len(metaData.loc[p]['query']) > 0):
                    self.product2query[p] = metaData.loc[p]['query']
                    words.update(' '.join(metaData.loc[p]['query']).split(' '))
            except:
                pass
            self.id2product[pid] = p
            self.product2id[p] = pid
            pid += 1
            
        self.productNum = pid
        self.nes_weight = np.zeros(self.productNum)
        self.queryNum = len(self.product2query)
        
        wi = 0
        self.word2id['<pad>'] = wi
        self.id2word[wi] = '<pad>'
        wi += 1
        for w in words:
            if(w==''):
                continue
            self.word2id[w] = wi
            self.id2word[wi] = w
            wi += 1
        self.wordNum = wi
        self.word_weight = np.zeros(wi)
    def init_dataset(self, reviewData,weights=True):
        try:
            self.data_X = []
            for r in range(len(reviewData)):
                if r % 100 == 0:
                    with open (r'out.txt','a+') as ff:
                        ff.write(str(len(reviewData))+ ' review: '+str(r) + '\n')
                rc = reviewData.iloc[r]
                try:
                    uid = self.user2id[rc['reviewerID']]
                    pid_pos = self.product2id[rc['asin']]
                    time_bin_pos = int(rc['timeBin'])
                except:
                    # 这个user没有加入到字典，购买次数不到3次
                    continue

                text = rc['reviewText']

                try:
                    # 得到product的query数组
                    q_text_array_pos = self.product2query[self.id2product[pid_pos]]
                except:
                    '''
                    没有对应的query
                    '''
                    continue
                try:
                    text_ids, len_r= self.trans_to_ids(text, self.max_review_len)
                    # 设置product的负采样频率
                    self.nes_weight[pid_pos] += 1
                except:
                    continue
                # 遍历每个物品的每个query 得到一个(u, p, q, r)元组
                for qi in range(len(q_text_array_pos)):
                    try:
                        qids_pos, len_pos = self.trans_to_ids(q_text_array_pos[qi], self.max_query_len)
                    except:
                        break
                    self.data_X.append((uid, pid_pos, qids_pos, len_pos, text_ids, len_r, time_bin_pos))
                    try:
                        self.userReviewsCount[uid] += 1
                        self.userReviewsCounter[uid] += 1
                    except:
                        self.userReviewsCount[uid] = 1
                        self.userReviewsCounter[uid] = 1


            '''
            数据集合划分 ---> 取每个用户购买过的item的最后一个
            '''
            for r in self.data_X:
                # 只考虑有3个以上（uqi）三元组的user
                if self.userReviewsCount[r[0]] > 2:
                    t = self.userReviewsCounter[r[0]]
                    if (t == 0):
                        continue
                    elif (t == 2): # 倒数第二个
                        self.eval_data.append(r)
                    elif (t == 1): # 倒数第一个
                        self.test_data.append(r)
                    else:
                        self.train_data.append(r)
                        self.time_data[r[6]].append(r)
                    self.userReviewsCounter[r[0]] -= 1

            if weights is not False:
                wf = np.power(self.nes_weight, 0.75)
                wf = wf / wf.sum()
                self.weights = wf
                wf = np.power(self.word_weight, 0.75)
                wf = wf / wf.sum()
                self.word_weight = wf
        except e:
            with open (r'out.txt','a+') as ff:
                ff.write(str(e)+ '\n')
    def trans_to_ids(self, query, max_len, weight_cal = True):
        query = query.split(' ')
        qids = []
        for w in query:
            if w == '':
                continue
            qids.append(self.word2id[w])
            # 需要统计词频
            if weight_cal:
                self.word_weight[self.word2id[w]-1] += 1
        for _ in range(len(qids), max_len):
            qids.append(self.word2id['<pad>'])
        return qids, len(query)
    
    def neg_sample(self):
        neg_item = []
        neg_word = []
        for ii in range(self.neg_sample_num):
            neg_item.append(self.sample_table_item[np.random.randint(self.table_len_item)])
            neg_word.append(self.sample_table_word[np.random.randint(self.table_len_word)])
        return neg_item,neg_word
    
    def init_sample_table(self):
        table_size = 1e6
        count = np.round(self.weights*table_size)
        self.sample_table_item = []
        for idx, x in enumerate(count):
            self.sample_table_item += [idx]*int(x)
        self.table_len_item = len(self.sample_table_item)
        
        count = np.round(self.word_weight*table_size)
        self.sample_table_word = []
        for idx, x in enumerate(count):
            self.sample_table_word += [idx]*int(x)
        self.table_len_word = len(self.sample_table_word)
    
    def __getitem__(self, i):
        pos = self.train_data[i]
        neg = self.neg_sample()
        
        return pos, neg
    def get_time_data(self, time_bin, i):
        pos = self.time_data[time_bin][i]
        neg = self.neg_sample()
        return pos, neg
    def getTestItem(self, i):
        return self.test_data[i]
    def __len__(self):
        return len(self.train_data)