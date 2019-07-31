from __future__ import print_function  # 这个是python当中让print都以python3的形式进行print，即把print视为函数
import torch  # 以下这几行导入相关的pytorch包
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

epochs = 10#跑的轮数
train_batch_size =  10 #batch的大小
test_batch_size = 1000 #测试的分batch的大小
log_interval = 10 #间隔记录
LR = 0.01#学习率
SDG_momentum = 0.5 #SGD参数
seed = 1 #随机
#定义数据其的采集和处理
temp_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]) #定义数据格式，进行归一化
train_set = datasets.MNIST('./data', train=True, download=True,transform=temp_transform) #输入数据的下载
train_loader = torch.utils.data.DataLoader(train_set,batch_size=train_batch_size,shuffle=True)
test_set = datasets.MNIST('./data', train=False, download=True, transform=temp_transform) #输入数据的改变
test_loader = torch.utils.data.DataLoader(test_set ,batch_size=train_batch_size,shuffle=True)
loss_train_graph = []
loss_validation_graph = []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.avg_pool2d(self.conv1(x), 2))
        x = F.relu(F.avg_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()  # 实例化一个网络对象
optimizer = optim.SGD(model.parameters(), lr = LR, momentum = SDG_momentum)  # 初始化优化器 model.train()
#optimizer = optim.Adam(model.parameters(), lr = LR,)  # 初始化优化器 model.train()

def train(epoch):  # 定义每个epoch的训练细节
    model.train()  # 设置为trainning模式
    temp_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(batch_idx)
        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
        loss = F.nll_loss(output, target)  # 计算损失函数
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 结束一次前传+反传之后，更新优化器参数
        if batch_idx % log_interval == 0:  # 准备打印相关信息，
            print('Train Epoch num: {} [{}/{}] \t tempLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), loss.data))
        temp_loss += loss.data
    temp_loss = temp_loss / len(train_loader)
    loss_train_graph.append(temp_loss)


def test():
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        temp_loss_b =  F.nll_loss(output, target, size_average=False)
        test_loss += temp_loss_b# sum up batch loss 把所有loss值进行累加
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nAverage loss:{:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    loss_validation_graph.append(test_loss)

if __name__ == '__main__':
    for epoch in range(1, epochs + 1):  # 以epoch为单位进行循环
        train(epoch)
        test()
    #画出图
    plt.figure(1)
    plt.plot(loss_train_graph)
    plt.show()
    plt.figure(2)
    plt.plot(loss_validation_graph)
    plt.show()