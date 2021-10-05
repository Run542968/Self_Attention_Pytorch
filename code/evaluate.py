import torch
from torch.autograd import Variable
from utils import visualize_attention

def evaluate(model,test_loader,word_to_id):
    model.eval()
    n_batches = 0
    correct = 0

    for batch_idx, train in enumerate(test_loader):
        model.hidden_state = model.init_hidden()
        x, y = Variable(train[0]), Variable(train[1])
        y_pred, att = model(x)#y_pred:[512,1,1],att:[512,10,200]

        correct += torch.eq(torch.max(y_pred.type(torch.DoubleTensor),dim=1).indices+1, y).data.sum()
        n_batches += 1
        #print("Batch accuracy{}:".format(torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)), y).data.sum()/test_loader.batch_size))
    print("Evaluate accuracy of the model", correct / (n_batches * test_loader.batch_size))

    # ##可视化
    iter_data=iter(test_loader)
    sentence=next(iter_data)[0]
    visualize_len=10
    visualize_sentence=sentence[:visualize_len]
    x=Variable(visualize_sentence)
    model.batch_size = x.size(0)
    model.hidden_state = model.init_hidden()
    _,wts = model(x)
    visualize_attention(wts,visualize_sentence.numpy(),word_to_id,filename='heatmap.html')


def evaluate_lstm(model,test_loader):
    model.eval()
    n_batches = 0
    correct = 0

    for batch_idx, train in enumerate(test_loader):
        model.hidden_state = model.init_hidden()
        x, y = Variable(train[0]), Variable(train[1])
        y_pred = model(x)#y_pred:[512,1,1],att:[512,10,200]

        correct += torch.eq(torch.max(y_pred.type(torch.DoubleTensor),dim=1).indices+1, y).data.sum()
        n_batches += 1
        #print("Batch accuracy{}:".format(torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)), y).data.sum()/test_loader.batch_size))
    print("Evaluate accuracy of the model", correct / (n_batches * test_loader.batch_size))


def evaluate_cnn(model,test_loader):
    model.eval()
    n_batches = 0
    correct = 0

    for batch_idx, train in enumerate(test_loader):
        x, y = Variable(train[0]), Variable(train[1])
        y_pred = model(x)#y_pred:[512,1,1],att:[512,10,200]

        correct += torch.eq(torch.max(y_pred.type(torch.DoubleTensor),dim=1).indices+1, y).data.sum()
        n_batches += 1
        #print("Batch accuracy{}:".format(torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)), y).data.sum()/test_loader.batch_size))
    print("Evaluate accuracy of the model", correct / (n_batches * test_loader.batch_size))