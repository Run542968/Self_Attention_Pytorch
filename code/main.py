from evaluate import evaluate,evaluate_lstm,evaluate_cnn
from dataset import DataSet
from model import SelfBiLSTMAttention,CNN_Pooling_MLP,BiLSTM_Pooling
from train import train,train_cnn,train_lstm
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def Attention_model(train_loader,test_loader,word2id,r):
    #加载模型
    model=SelfBiLSTMAttention(embedding_dim=100,
                                        lstm_hide_dim=1000,
                                        max_time_len=100,
                                        batch_size=train_loader.batch_size,
                                        da=350,
                                        r=r,
                                        class_num=5,
                                        word2id=word2id,
                                        use_pre_embeddings=True,
                                        pre_embeddings_path="glove.6B/glove.6B.100d.txt")
    print(model)
    #损失函数
    loss=torch.nn.CrossEntropyLoss()
    #优化器
    optimizer=torch.optim.Adam(model.parameters())
    #开始训练
    _, batch_acc,trained_model=train(model=model,
                                         train_loader=train_loader,
                                         criterion=loss,
                                         optimizer=optimizer,
                                         epochs=1,
                                         use_regularization=True,
                                         C=0.06,
                                         clip=True)
    # #开始测试
    evaluate(model,test_loader,word2id)
    return batch_acc

def plot_curve(train_loader, test_loader, word2id):
    acc_list = []
    r = [1, 10, 20, 30]
    for i in r:
        acc = Attention_model(train_loader, test_loader, word2id, i)
        acc_list.append(acc)
    acc_list = np.array(acc_list)

    acc_list1 = []
    for l in acc_list:
        temp = []
        for i in range(len(l)):
            if (i % 7 == 0):
                temp.append(l[i])
        acc_list1.append(temp)

    x = np.linspace(0, 40, 41)
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    for l in acc_list1:
        plt.plot(x, l)
    plt.legend(['r=1', 'r=10', 'r=20', 'r=30'], loc='upper left')
    plt.show()

def BiLSTM_model(train_loader,test_loader,word2id):
    model = BiLSTM_Pooling(word2id=word2id,
                           embedding_dim=100,
                           batch_size=train_loader.batch_size,
                           lstm_hide_dim=300,
                           max_time_len=100,
                           class_num=5)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    losses, accuracy, trained_model = train_lstm(model=model,
                                                 train_loader=train_loader,
                                                 criterion=loss,
                                                 optimizer=optimizer,
                                                 epochs=1,
                                                 clip=True)
    evaluate_lstm(trained_model, test_loader=test_loader)

def CNN_model(train_loader,test_loader,word2id):
    model=CNN_Pooling_MLP(word2id=word2id,
                          embeddings_dim=100,
                          batch_size=train_loader.batch_size,
                          sqe_len=100,
                          class_num=5)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    losses, accuracy,trained_model=train_cnn(model=model,
                                             train_loader=train_loader,
                                             criterion=loss,
                                             optimizer=optimizer,
                                             epochs=1,
                                             clip=True)
    evaluate_cnn(trained_model, test_loader=test_loader)
if __name__ == "__main__":
    #加载数据
    dataset=DataSet('yelp_dataset/data_process/word2id.pkl',
                    'yelp_dataset/data_process/dataset_list_20000num.pkl',
                    'yelp_dataset/data_process/dataset_label_20000.pkl',
                    train_size=18000,
                    test_size=2000,
                    batch_size=64,
                    squ_len=100)
    train_loader,test_loader,word2id=dataset.data_loader_pre()

    #attention_model
    Attention_model(train_loader,test_loader,word2id,r=30)
    #plot
    #plot_curve(train_loader,test_loader,word2id)

    #BiLSTM+MaxPooling+MLP
    #BiLSTM_model(train_loader,test_loader,word2id)

    #CNN+MaxPooling+MLP
    #CNN_model(train_loader,test_loader,word2id)


