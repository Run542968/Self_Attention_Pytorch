import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional  as F
import numpy as np


class SelfBiLSTMAttention(nn.Module):
    def __init__(self,embedding_dim,lstm_hide_dim,max_time_len,batch_size,da,r,class_num,word2id,use_pre_embeddings=False,pre_embeddings_path=None):
        super(SelfBiLSTMAttention, self).__init__()
        ##初始embedding
        self.word2id=word2id
        self.embeddings,embeddings_dim=self.load_embedding(use_pre_embeddings,pre_embeddings_path,word2id,embedding_dim)

        ##BiLSTM参数及构建
        self.num_layer=1
        self.batch_first=True
        self.bidirectional=True
        self.max_time_len = max_time_len
        self.lstm_hide_dim=lstm_hide_dim
        self.lstm_input=embedding_dim
        self.batch_size=batch_size
        self.hidden_state=self.init_hidden()
        self.bilstm=nn.LSTM(input_size=embedding_dim,hidden_size=lstm_hide_dim,num_layers=self.num_layer,batch_first=self.batch_first,bidirectional=self.bidirectional)

        #self-attention参数及构建
        self.da=da
        self.r=r
        self.class_num=class_num
        self.Ws1=nn.Linear(in_features=2*lstm_hide_dim,out_features=da)
        self.Ws2=nn.Linear(in_features=da,out_features=r)
        self.fc=nn.Linear(in_features=lstm_hide_dim*2,out_features=class_num)

    def load_embedding(self,use_pre_embeddings,pre_embeddings_path,word2id,embeddings_dim):
        if not use_pre_embeddings:
            word_embeddings = torch.nn.Embedding(len(word2id), embeddings_dim, padding_idx=0)
        elif use_pre_embeddings:
            embeddings = np.zeros((len(word2id), embeddings_dim))
            with open(pre_embeddings_path,encoding="utf-8") as f:
                for line in f.readlines():
                    values = line.split()
                    word = values[0]
                    index = word2id.get(word)
                    if index:
                        vector = np.array(values[1:], dtype='float32')
                        if vector.shape[-1] != embeddings_dim:
                            raise Exception('Dimension not matching.')
                        embeddings[index] = vector
            pre_embeddings=torch.from_numpy(embeddings).float()
            word_embeddings = torch.nn.Embedding(pre_embeddings.size(0), pre_embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(pre_embeddings)
        return word_embeddings, embeddings_dim

    def init_hidden(self):
        bidirection=2 if self.bidirectional else 1
        return (Variable(torch.zeros(self.num_layer*bidirection,self.batch_size,self.lstm_hide_dim)),
                Variable(torch.zeros(self.num_layer*bidirection,self.batch_size,self.lstm_hide_dim)))

    def attention(self,H):#H:[bs,n,2u]
        x = torch.tanh(self.Ws1(H))#x:[bs,n,da]
        x = self.Ws2(x)#x:[bs,n,r]
        # x = self.softmax(x, 1)#x:[512,200,10]
        x=F.softmax(x,dim=1)
        # print("aftersoftmax:",x)
        # print("aftersoftmax.shape",x.shape)
        A = x.transpose(1, 2)#A:[bs,r,n]
        return A

    def forward(self,x):#x:[bs,n]
        embeddings=self.embeddings(x)#embeddings[bs,n,embeddings_dim]
        #print(embeddings.size())
        H,self.hidden_state=self.bilstm(embeddings.view(self.batch_size, self.max_time_len, -1),self.hidden_state)#H:[bs,n,2u]
        A=self.attention(H)#A:[bs,r,n]
        M = A @ H  # 这里的装饰器就是矩阵乘法,M:[bs,r,2u]
        avg_sentence_embeddings = torch.sum(M, 1) / self.r#[bs,2u]
        output=self.fc(avg_sentence_embeddings)
        return output, A

    def l2_matrix_norm(self, m):
        return torch.sum(torch.sum(torch.sum(m ** 2, 1) ** 0.5).type(torch.DoubleTensor))


class CNN_Pooling_MLP(nn.Module):
    def __init__(self,word2id,embeddings_dim,batch_size,class_num=1,hide_dim=300,sqe_len=100):
        super(CNN_Pooling_MLP, self).__init__()
        self.embeddings = torch.nn.Embedding(len(word2id), embeddings_dim, padding_idx=0)
        self.hide_dim=hide_dim
        self.sqe_len=sqe_len
        self.class_num=class_num
        self.batch_size=batch_size
        self.conv=nn.Conv1d(in_channels=embeddings_dim,out_channels=hide_dim,kernel_size=3,stride=1)
        self.max_pooling=nn.MaxPool1d((sqe_len-3)+1)
        self.fc1=nn.Sequential(
            nn.Linear(in_features=hide_dim,out_features=3000),
        )
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Sequential(
            nn.Linear(in_features=3000,out_features=class_num),
        )

    def forward(self,x):#x:[bs,sqe_len]
        emb=self.embeddings(x)#emb:[bs,squ_len,emb_dim]
        # conv_x=emb.unsqueeze(1)#conv_x:[bs,1,squ_len,emb_dim]
        # x=self.conv(conv_x)#x:[bs,hide_dim,-,1]
        # x=x.squeeze(3)#x:[bs,hide_dim,-]
        # x=self.max_pooling(x)#x:[bs,hide_dim,1]
        # x=x.squeeze(2)#x:[bs,hide_dim]
        # x=self.fc1(x)#x:[bs,3000]
        # x=self.dropout(x)#x:[bs,3000]
        # x=self.fc2(x)#x:[bs,1]
        # out=torch.sigmoid(x)#x:[bs,1]
        conv_x=emb.permute(0,2,1)#conv_x:[bs,emb_dim,seq_len]
        x=self.conv(conv_x)#x:[bs,hide_dim,(seq_len-3)/1+1]
        x=self.max_pooling(x)
        x=x.squeeze(2)
        x=self.fc1(x)
        x=self.dropout(x)
        out=self.fc2(x)
        return out

class BiLSTM_Pooling(nn.Module):
    def __init__(self,word2id,embedding_dim,lstm_hide_dim,max_time_len,batch_size,class_num):
        super(BiLSTM_Pooling, self).__init__()
        ##初始embedding
        self.embeddings= torch.nn.Embedding(len(word2id), embedding_dim, padding_idx=0)
        self.embeddings_dim=embedding_dim

        ##BiLSTM参数及构建
        self.num_layer=1
        self.batch_first=True
        self.bidirectional=True
        self.max_time_len = max_time_len
        self.lstm_hide_dim=lstm_hide_dim
        self.batch_size=batch_size
        self.hidden_state=self.init_hidden()
        self.bilstm=nn.LSTM(input_size=embedding_dim,hidden_size=lstm_hide_dim,num_layers=self.num_layer,batch_first=self.batch_first,bidirectional=self.bidirectional)

        self.max_pooling=nn.MaxPool1d(kernel_size=max_time_len)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=2*lstm_hide_dim, out_features=3000),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=3000, out_features=class_num),
        )


    def init_hidden(self):
        bidirection=2 if self.bidirectional else 1
        return (Variable(torch.zeros(self.num_layer*bidirection,self.batch_size,self.lstm_hide_dim)),
                Variable(torch.zeros(self.num_layer*bidirection,self.batch_size,self.lstm_hide_dim)))


    def forward(self,x):#x:[bs,n],n=sqe_len
        embeddings=self.embeddings(x)
        H,self.hidden_state=self.bilstm(embeddings.view(self.batch_size, self.max_time_len, -1),self.hidden_state)#H:[bs,n,2u]
        H=torch.transpose(H,1,2)#H:[bs,2u,n]
        x=self.max_pooling(H)#x:[bs,2u,1]
        x=x.squeeze(2)#x:[bs,2u]
        x=self.fc1(x)#x:[bs,3000]
        x=self.dropout(x)#x:[bs,3000]
        out=self.fc2(x)#x[bs,5]
        return out
