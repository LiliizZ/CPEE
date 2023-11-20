import logging
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import os
import json
from gensim.models.word2vec import Word2Vec
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from argments import parser
from tqdm import tqdm
import time

from load_file import get_time_dif

logger = logging.getLogger(__file__)

dict_cause = {"无": 0, "民间借贷纠纷": 1, "金融借款合同纠纷": 2,
              "商品房预售合同纠纷": 3, "商品房销售合同纠纷": 4,
              "借款合同纠纷": 5, "房屋买卖合同纠纷": 6,
              "租赁合同纠纷": 7, "买卖合同纠纷": 8,
              "物业服务合同纠纷": 9, "劳务合同纠纷": 10
              }

def ex_shuju(xx):
    xx = xx.replace('第', '')
    xx = xx.replace('、', '')
    xx = xx.replace('百零', '10')
    xx = xx.replace('零', '')
    xx = xx.replace('条', '')
    xx = xx.replace('百', '100')
    return xx


class CPEEData:
    def __init__(self, args):
        self.label_dict = {'contradiction': 0, 'entailment': 1, 'neutral': 2, 'non-entailment': 0}
        self.args = args
        #self.max_length = args.max_length
        self.batch_size = args.batch_size

        self.train_data_path = os.path.join("/data/liliz/hecp/data/cpee/", 'train.json')
        self.val_data_path = os.path.join("/data/liliz/hecp/data/cpee/", 'val.json')
        self.test_data_path = os.path.join("/data/liliz/hecp/data/cpee/", 'test.json') #all tokenized
   
        self.word2vec_model = Word2Vec.load("/data/liliz/hecp/word2vec.model")
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_text = []
        self.train_label = []

        self.gen_emb = np.load(f"/data/liliz/hecp/article_title/articles_emb/art_gen_gt.npy")
        self.cause_all = np.load(f"/data/liliz/hecp/new_single_np/val/val_cause_labels.npy")
        self.cause4_all = np.load(f"/data/liliz/hecp/new_single_np/val/val_cause4_labels.npy")
        self.cause3_all = np.load(f"/data/liliz/hecp/new_single_np/val/val_cause3_labels.npy")

        self.tokenizer = Tokenizer()


        #self.init_data()

    def init_data(self):
        #self.train_data = self.generator(self.train_data_path, "train")
        #self.val_data = self.generator(self.val_data_path, "val")
        self.test_data = self.generator(self.test_data_path, "test")

    def get_word_vector(self, token):
        # 如果token在词汇表中，返回对应的词向量；否则返回一个默认向量（可以根据具体情况调整）
        return self.word2vec_model[token] if token in self.word2vec_model else [0.0] * self.word2vec_model.vector_size

    def get_spe_vector(self, cause):
        if cause == 1 or cause == 2 or cause == 5:#196-211
            spe_gt = np.load(f"/data/liliz/hecp/article_title/articles_emb/art_spe_196.npy")
            
        elif cause == 3 or cause == 4 or cause == 6 or cause == 8:  # 130-175
            spe_gt = np.load(f"/data/liliz/hecp/article_title/articles_emb/art_spe_130.npy")
            
        elif cause == 7: #212-236
            spe_gt = np.load(f"/data/liliz/hecp/article_title/articles_emb/art_spe_212.npy")
            
        elif cause == 9: #176-184
            spe_gt = np.load(f"/data/liliz/hecp/article_title/articles_emb/art_spe_176.npy")
            
        elif cause == 10:  # 251-268
            spe_gt = np.load(f"/data/liliz/hecp/article_title/articles_emb/art_spe_251.npy")

        return spe_gt



    def generator(self, data_path, data_type):
        #if os.path.exists(f"./test/{data_type}_result.npy"):
            #result = np.load(f"./test/{data_type}_result.npy", allow_pickle=True)
        #else:
        start_time = time.time()
        result = []
        Fact = []
        Plai = []
        Defe = []
        Plea = []
        Cause_emb = []
        Cause4_emb = []
        Cause3_emb = []
        Specific_emb = []
        General_emb = []
        Label_plea = []
        Label_cause = []
        Label_cause4 = []
        Label_cause3 = []
        Label_specific = []
        Label_general = []
        id = 0
        f = open(data_path, 'r')
        lines = json.load(f)
        print("总共", len(lines))
        tokenizer = Tokenizer()
        for line in tqdm(lines, desc='Data Preprocess~'):
            #try:
            if line['plai'] == None:
                line['plai'] = "无"
            if line['defe'] == None:
                line['defe'] = "无"
            article1 = []
            article2 = []
            for item in line['article']:
                item = ex_shuju(item)
                if "百" in item:
                    continue
                if item == "" or int(item) > 430:
                    continue
                if 0 <= int(item) <= 129:
                    # line['article'].append(item)
                    article1.append(item)
                else:
                    article2.append(item)

            #for item in line['plea']:   
            for i in range(0, len(line['plea'])): 
                id += 1
                fact = line['fact']
                plai = line['plai']
                defe = line['defe']
                plea = line['plea'][i]
                label_plea = line['label'][i]
                label_cause = dict_cause[line['cause'][0]]

                if label_cause > 4:
                    a = 0
                    label_cause4 = a 
                    label_cause3 = label_cause - 5
    
                else:
                    label_cause4 = label_cause
                    if label_cause == 1 or label_cause == 2:
                        label_cause3 = 1
                    if label_cause == 3 or label_cause == 4:
                        label_cause3 = 2

                if len(article1) == 0:
                    article1 = ['0']
                if len(article2) == 0:
                    article2 = ['0']  
                label_article = line['article']
                label_general =  article1
                label_specific = article2    

                specific_emb = self.get_spe_vector(label_cause)  
                general_emb =  self.gen_emb

                cause_emb = self.cause_all[1, :, :]
                cause4_emb = self.cause4_all[1, :, :]
                cause3_emb = self.cause3_all[1, :, :] 
        

                '''result.append(id, fact, plai, defe, plea, 
                                cause_emb, cause4_emb, cause3_emb,
                                specific_emb, general_emb,
                                label_plea, label_cause, label_cause4, label_cause3, 
                                label_article, label_general, label_specific)'''
                Fact.append(fact)
                Plai.append(plai)
                Defe.append(defe)
                Plea.append(plea)
                Cause_emb.append(cause_emb)
                Cause4_emb.append(cause4_emb)
                Cause3_emb.append(cause3_emb)
                Specific_emb.append(specific_emb)
                General_emb.append(general_emb)
                Label_cause.append(label_cause)
                Label_cause4.append(label_cause4)
                Label_cause3.append(label_cause3)
                Label_specific.append(label_specific)
                Label_general.append(label_general)
                Label_plea.append(label_plea)


        print("Fact----------")
        tokenizer.fit_on_texts(Fact)
        fact_sequences = tokenizer.texts_to_sequences(Fact)
        fact_test = np.array(fact_sequences)
        print("fact_sequence.shape", fact_test.shape)
        facts_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            fact_sequences, maxlen=200, dtype='int32', padding="pre", truncating="pre", value=0)
        facts_sequences = np.array(facts_sequences)
        print("fact", facts_sequences, facts_sequences.shape)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_fact.npy", facts_sequences)

        print("Plai-----------")
        tokenizer.fit_on_texts(Plai)
        plai_sequences = tokenizer.texts_to_sequences(Plai)
        plai_test = np.array(plai_sequences)
        print("plai_sequence.shape", plai_test.shape)
        plais_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            plai_sequences, maxlen=100, dtype='int32', padding="pre", truncating="pre", value=0)
        plais_sequences = np.array(plais_sequences)
        print("plai", plais_sequences, plais_sequences.shape)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_plai.npy", plais_sequences)

        print("Defe-----------")
        tokenizer.fit_on_texts(Defe)
        defe_sequences = tokenizer.texts_to_sequences(Defe)
        defe_test = np.array(defe_sequences)
        print("defe_sequence.shape", defe_test.shape)
        defes_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            defe_sequences, maxlen=50, dtype='int32', padding="pre", truncating="pre", value=0)
        defes_sequences = np.array(defes_sequences)
        print("defe", defes_sequences, defes_sequences.shape)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_defe_50.npy", defes_sequences)

        print("Plea-----------")
        tokenizer.fit_on_texts(Plea)
        plea_sequences = tokenizer.texts_to_sequences(Plea)
        plea_test = np.array(plea_sequences)
        print("plea_sequence.shape", plea_test.shape)
        pleas_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            plea_sequences, maxlen=50, dtype='int32', padding="pre", truncating="pre", value=0)
        pleas_sequences = np.array(pleas_sequences)
        print("plea", pleas_sequences, pleas_sequences.shape)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_plea_50.npy", pleas_sequences)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_cause_emb.npy", Cause_emb)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_cause4_emb.npy", Cause4_emb)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_cause3_emb.npy", Cause3_emb)

        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_specific_emb.npy", Specific_emb)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_general_emb.npy", General_emb)
       
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_label_cause.npy", Label_cause)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_label_cause4.npy", Label_cause4)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_label_cause3.npy", Label_cause3)

        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_label_specific.npy", Label_specific)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_label_general.npy", Label_general)
        np.save(f"/share/liliz/CPEE/cpee_data/{data_type}/{data_type}_label_plea.npy", Label_plea)

        time_dif = get_time_dif(start_time)
        print("loading time: ", time_dif)
            
        print(f"len", len(Fact))
        return result


    
    

    def _preprocess(self, fact, plai, defe, plea, cause_emb, cause4_emb, cause3_emb, specific_emb, general_emb, label_plea, label_cause, label_cause4, label_cause3, label_specific, label_general):
        Fact = tf.convert_to_tensor([self.word2vec_embedding(word_list) for word_list in fact.numpy()])
        Plai = tf.convert_to_tensor([self.word2vec_embedding(word_list) for word_list in plai.numpy()])
        Defe = tf.convert_to_tensor([self.word2vec_embedding(word_list) for word_list in defe.numpy()])
        Plea = tf.convert_to_tensor([self.word2vec_embedding(word_list) for word_list in plea.numpy()])
        return (Fact, Plai, Defe, Plea, cause_emb, cause4_emb, cause3_emb, specific_emb, general_emb, 
                label_plea, label_cause, label_cause4, label_cause3, label_specific, label_general)

    def get_tensor_slices(self, data):
        dataset = self.get_dataset(data)
        return dataset.map(self.collate_fn)
    

    def word2vec_embedding(self, word_list):
        # 如果词汇表中的词在Word2Vec模型中不存在，则用零向量表示
        return np.array([self.word2vec_model[word] if word in self.word2vec_model else np.zeros_like(self.word2vec_model.wv['1']) for word in word_list])


    def get_dataloader(self, data, batch_size, shuffle=True):
        #dataset = self.get_dataset(data)
        '''data = self.collate_fn(data)
        dataset = tf.data.Dataset.from_tensor_slices(data)
        
        dataset = dataset.batch(batch_size).repeat(num_epochs)'''
        
        dataset = tf.data.Dataset.from_generator(
            lambda:data,
            (tf.int32, tf.int32, tf.int32, tf.int32,
                 tf.int32, tf.int32, tf.int32, tf.int32,  # cl, cl4 cl3, gl
                 tf.int32, tf.int32, tf.int32, tf.int32,  # sl, c, c4, c3
                 tf.int32, tf.int32, tf.int32)  # g, s, j
            ).padded_batch(batch_size,
                           padded_shapes=(
                               tf.TensorShape([None]), # fact len
                               tf.TensorShape([None]), tf.TensorShape([None]),  tf.TensorShape([None]), # plea, 
                               tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                               tf.TensorShape([None, None]), tf.TensorShape([None, None]), 
                               tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]),
                               tf.TensorShape([None]), tf.TensorShape([None]), 
                               tf.TensorShape([None])),
                           padding_values=(0, 0, 0, 0, 
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data))
  
       
        # 数据预处理和批处理
        
        return dataset
    
    
args = parser()
cpee_dataset = CPEEData(args)
test_dataset = cpee_dataset.generator(cpee_dataset.test_data_path, "test")
val_dataset = cpee_dataset.generator(cpee_dataset.val_data_path, "val")
train_dataset = cpee_dataset.generator(cpee_dataset.train_data_path, "train")





