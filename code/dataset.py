import spacy
import torch
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import torch.utils.data as data_utils
import pandas as pd
import pickle

class DataSet():
    def __init__(self,word2id_path,dataset_list_path,dataset_label_path,train_size,test_size,squ_len,batch_size=32):
        self.word2id_path=word2id_path
        self.dataset_list_path=dataset_list_path
        self.dataset_label_path=dataset_label_path
        self.batch_size=batch_size
        self.train_size=train_size
        self.test_size=test_size
        self.squ_len=squ_len

    def data_loader_pre(self):
        '''
        加载已经保存好的数据
        :return: dataLoader和word2id
        '''
        with open(self.dataset_list_path,'rb') as file:
            dataset_list=pickle.load(file)
        with open(self.dataset_label_path,'rb') as file:
            dataset_label=pickle.load(file)
        with open(self.word2id_path,'rb') as file:
            word2id=pickle.load(file)
        return self.data_loader(dataset_list,dataset_label,word2id,batch_size=self.batch_size,squ_len=self.squ_len)

    def data_loader(self,dataset_list,dataset_label,word2id,batch_size,squ_len):
        '''
        完成数据的训练集/测试集划分，完成词向量的补齐,最终返回data_loader迭代器
        :param dataset_list: 在上一步被转化为numpy的数据集
        :param dataset_label:对应的标签
        :param word2id: 词表
        :param batch_size: 构建dataloader迭代器时的batchsize
        :param squ_len: 文本的长度，不够的用PAD补全
        :return: 数据的dataLoader迭代器
        '''
        #划分训练集/测试集
        state = np.random.get_state()
        np.random.shuffle(dataset_list)
        np.random.set_state(state)
        np.random.shuffle(dataset_label)
        train_x, test_x = dataset_list[:self.train_size],dataset_list[self.train_size:]
        train_y, test_y = dataset_label[:self.train_size],dataset_label[self.train_size:]
        #补齐句子长度
        train_x_pad = pad_sequences(train_x, maxlen=squ_len)
        test_x_pad = pad_sequences(test_x, maxlen=squ_len)
        #把数据格式转为tensor，并打包成（x,y）格式
        train_data = data_utils.TensorDataset(torch.from_numpy(train_x_pad).type(torch.LongTensor),
                                              torch.from_numpy(train_y).type(torch.DoubleTensor))
        test_data=data_utils.TensorDataset(torch.from_numpy(test_x_pad).type(torch.LongTensor),
                                           torch.from_numpy(test_y).type(torch.DoubleTensor))
        #把打包好的数据，封装进dataLoader迭代器
        train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, drop_last=True)
        test_loader=data_utils.DataLoader(test_data,batch_size=batch_size,drop_last=True)
        return train_loader,test_loader,word2id

    def get_dataset_dic(self,tran_data_path,test_data_path):
        '''
        获取数据集的【词表】以及【清洗过的数据】，并保存
        :param tran_data_path: 训练集数据路径
        :param test_data_path: 测试集数据路径
        :return: word2id,id2word，对每一提条文本完成数据清晰的dataset：格式为[['i','am','a','pig',...],['you','are','so',...],....]
        '''
        tokenizer = spacy.load('en_core_web_sm')  # 加载英文分词器   *****注意：spacy和keras会有冲突
        dataset_list=list()#保存分词后的文本序列
        word_to_id=dict()
        id_to_word=list()
        test_data=pd.read_csv("yelp_dataset/data/test.csv",names=['star','review'])
        train_data=pd.read_csv("yelp_dataset/data/train.csv",names=['star','review'])
        dataset=pd.concat([train_data,test_data],axis=0)
        dataset=dataset.reset_index(drop=True)
        dataset_label=dataset.star.values
        word_to_id["<PAD>"]=0
        word_to_id["<UNK>"]=1
        count=2
        for i in range(len(dataset)):
            sentence=dataset.loc[i,'review']
            words=tokenizer(sentence)
            word_list=[token.orth_ for token in words if not token.is_punct | token.is_space | token.is_stop]
            dataset_list.append(word_list)
            for word in word_list:
                word_to_id[word]=count
                count=count+1
            print("The {}/{} has been processed. ".format(i,len(dataset)))

        for k, v in word_to_id.items():
            id_to_word.insert(v, k)

        word_to_id={k:v for v,k in enumerate(id_to_word)}
        #保存词表
        with open('yelp_dataset/data_process/word2id.pkl','wb') as file:
            pickle.dump(word_to_id,file)
        with open('yelp_dataset/data_process/id2word.pkl','wb') as file:
            pickle.dump(id_to_word,file)
        with open('yelp_dataset/data_process/dataset_list.pkl','wb') as file:
            pickle.dump(dataset_list,file)
        return word_to_id,id_to_word,dataset_list,dataset_label

    def text2num(self,dataset_list,word2id):
        '''
        利用获取到的词表将原始英文文本数据集转换为数字序列numpy形式
        :param dataset_list: 经过数据清洗的文本集，格式为[['i','am','a','pig',...],['you','are','so',...],....]
        :param word2id: 已经计算完成的词表，是一个字典{'word':id}
        :return: 将文本转化为词表中标号，并转化为numpy
        '''
        for i in range(len(dataset_list)):
            for j in range(len(dataset_list[i])):
                word=dataset_list[i][j]
                if( word in word2id):
                    dataset_list[i][j]=word2id[word]
                else:
                    dataset_list[i][j]=1
        dataset_list=np.array(dataset_list)
        return dataset_list

if __name__ =="__main__":
    print("开心！")