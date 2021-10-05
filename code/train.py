import torch
from torch.autograd import Variable


def train(model, train_loader, criterion, optimizer, epochs=5, use_regularization=False, C=0, clip=False):
    model.train()
    losses = []
    accuracy = []
    batch_accuracy = []
    for i in range(epochs):
        print("Running EPOCH", i + 1)
        total_loss = 0
        n_batches = 0
        correct = 0
        for batch_idx, train in enumerate(train_loader):
            model.hidden_state = model.init_hidden()
            x, y = Variable(train[0]), Variable(train[1])
            y_pred, att = model(x)#y_pred:[512,1,1],att:[512,10,200]

            # penalization AAT - I
            if use_regularization:
                attT = att.transpose(1, 2)#attT:[512,200,10]
                identity = torch.eye(att.size(1))#eye()对角单位矩阵[10,10]
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size, att.size(1), att.size(1)))#把[10,10]copy了512份，变成[512,10,10]
                penal = model.l2_matrix_norm(att @ attT - identity)

            batch_acc=torch.eq(torch.max(y_pred.type(torch.DoubleTensor),dim=1).indices+1, y).data.sum()
            correct += batch_acc
            if use_regularization:
                loss = criterion(y_pred,y.long()-1)+ C * penal / train_loader.batch_size
            else:
                loss = criterion(y_pred, y.long()-1)

            total_loss += loss.data
            optimizer.zero_grad()#清空过往梯度
            loss.backward()#反向传播，计算当前梯度；

            # gradient clipping
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()#根据梯度，更新网络参数
            n_batches += 1
            batch_accuracy.append(batch_acc)
            print("Epoch: {} Batch: {} loss:{}".format(i,n_batches,loss))

        print("Epoch avg_loss is", total_loss / n_batches)
        print("Train accuracy of the model", correct / (n_batches * train_loader.batch_size))
        losses.append(total_loss / n_batches)
        accuracy.append(correct / (n_batches * train_loader.batch_size))

    return losses, batch_accuracy,model



def train_cnn(model, train_loader, criterion, optimizer, epochs=5, clip=False):
    model.train()

    losses = []
    accuracy = []
    for i in range(epochs):
        print("Running EPOCH", i + 1)
        total_loss = 0
        n_batches = 0
        correct = 0

        for batct_id, train in enumerate(train_loader):

            x, y = Variable(train[0]), Variable(train[1])
            y_pred = model(x)
            print("y_pred:{}",torch.max(y_pred.type(torch.DoubleTensor),dim=1).indices+1)
            print("y:{}",y)

            correct += torch.eq(torch.max(y_pred.type(torch.DoubleTensor),dim=1).indices+1, y).data.sum()
            loss = criterion(y_pred, y.long() - 1)

            total_loss += loss.data
            optimizer.zero_grad()#清空过往梯度
            loss.backward()#反向传播，计算当前梯度；

            # gradient clipping
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()#根据梯度更新网络参数
            n_batches += 1
            print("Epoch: {} Batch: {} loss:{}".format(i,n_batches,loss))

        print("Epoch avg_loss is", total_loss / n_batches)
        print("Train accuracy of the model", correct / (n_batches * train_loader.batch_size))
        losses.append(total_loss / n_batches)
        accuracy.append(correct / (n_batches * train_loader.batch_size))
    return losses, accuracy, model

def train_lstm(model, train_loader, criterion, optimizer, epochs=5, clip=False):
    model.train()
    losses = []
    accuracy = []
    for i in range(epochs):
        print("Running EPOCH", i + 1)
        total_loss = 0
        n_batches = 0
        correct = 0

        for batch_idx, train in enumerate(train_loader):
            model.hidden_state = model.init_hidden()
            x, y = Variable(train[0]), Variable(train[1])
            y_pred = model(x)#y_pred:[bs,5]
            print("y_pred:{}",torch.max(y_pred.type(torch.DoubleTensor),dim=1).indices+1)
            print("y:{}",y)
            correct += torch.eq(torch.max(y_pred.type(torch.DoubleTensor),dim=1).indices+1, y).data.sum()
            loss = criterion(y_pred, y.long() - 1)

            total_loss += loss.data
            optimizer.zero_grad()#清空过往梯度
            loss.backward()#反向传播，计算当前梯度；

            # gradient clipping
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()#根据梯度更新网络参数
            n_batches += 1
            print("Epoch: {} Batch: {} loss:{}".format(i,n_batches,loss))

        print("Epoch avg_loss is", total_loss / n_batches)
        print("Train accuracy of the model", correct / (n_batches * train_loader.batch_size))
        losses.append(total_loss / n_batches)
        accuracy.append(correct / (n_batches * train_loader.batch_size))
    return losses, accuracy,model