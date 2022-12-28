import datetime
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet34
from Create_dataLoader import LoadData
# from my_torch_model import Net
from model_ex import Net,VGG16,RestNet18,ResNet50,Bottleneck,ResNet34
from torch.optim.lr_scheduler import StepLR


class model_param_init(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert isinstance(model, nn.Module), 'model not a class nn.Module'
        self.net = model
        self.initParam()
    def initParam(self):
        for param in self.net.parameters():
            # nn.init.zeros_(param)
            # nn.init.ones_(param)
            nn.init.normal_(param, mean=0, std=0.5)
            # nn.init.uniform_(param, a=0, b=1)
            # nn.init.constant_(param, val=1)   # 将所有权重初始化为1
            # nn.init.eye_(param)  # 只能将二维的tensor初始化为单位矩阵
            # nn.init.xavier_uniform_(param, gain=1)  # Glorot初始化  得到的张量是从-a——a中采用的
            # nn.init.xavier_normal_(param, gain=1)   # 得到的张量是从0-std采样的
            #nn.init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='leaky_relu') # he初始化方法
            # nn.init.kaiming_uniform_(param)


# 定义训练函数，需要
def train(dataloader, model, loss_fn, optimizer, tac, myloss):
    size = len(dataloader.dataset)
    correct = 0
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        # 将数据存到显卡
        X, y = X.cuda(), y.cuda()
        X = X.to(torch.float32)
        # 得到预测的结果pred
        pred = model(X)
        # 计算预测的误差
        # print(pred,y)
        loss = loss_fn(pred, y)
        # 统计预测正确的个数
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每训练100次，输出一次当前信息
    loss, current = loss.item(), batch * len(X)
    print(f"tran_loss: {loss:>7f}")
    myloss.append(loss)
    correct /= size
    tac.append(correct)
    print(f"train_correct: {correct:>7f}")
    print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f} \n")

def te_mymodel(dataloader, model, my_t_ac, pre_label, real_label,loss_fn):
    size = len(dataloader.dataset)
    print("size = ", size)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            # 将数据转到GPU
            X, y = X.cuda(), y.cuda()
            X = X.to(torch.float32)
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            pre_label.append(pred.argmax(1))
            real_label.append(y)
            # 计算预测值pred和真实值y的差距
            # test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    # myloss.append(test_loss)
    my_t_ac.append(100 * correct)
    # print("test_correct = ", correct)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_my_model(epochs=1):
    batch_size = 128
    # # 给训练集和测试集分别创建一个数据集加载器
    train_data = LoadData("train.txt")
    valid_data = LoadData("test.txt")
    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size)

    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cuda"
    print("Using {} device".format(device))
    # model = Net()
    model = ResNet34()
    #model.apply(model_param_init)

    # model = resnet34(pretrained=True, progress=True)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(model.fc.in_features, len(os.listdir(base_path)))
    # 调用刚定义的模型，将模型转到设置好的device（如果可用）
    model.to(device)

    # print(model)

    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.L1Loss(reduction='mean')
    # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)  # 初始学习率
    scheduler_1 = StepLR(optimizer, step_size=30, gamma=0.2)
    # 一共训练epochs次
    epochs = epochs
    my_loss = []
    my_ac = []
    tac = []
    real_label = []
    pre_label = []
    st = datetime.datetime.now().strftime('%b%m%H%M%S')
    # with open(f"./data/pths_info/data{st}_model.info","w",encoding="utf8") as f:
    #     ls = os.listdir("./data/test_qx_data")
    #     f.write(",".join(ls))
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        time_now_1 = datetime.datetime.now()
        train(train_dataloader, model, loss_fn, optimizer, tac, my_loss)
        scheduler_1.step()
        print(f"lr:{scheduler_1.get_lr()}")
        real_label = []
        pre_label = []
        te_mymodel(test_dataloader, model, my_ac, pre_label, real_label,loss_fn)
        if ((t + 1) % 10 == 0) or ((t + 1) == epochs):
            torch.save(model.state_dict(), f"./mytorch_model/{st}_model_{t + 1}.pth")
            print("Saved PyTorch Model State to model.pth")
        time_now_2 = datetime.datetime.now()
        run_time = time_now_2 - time_now_1
        print('运行时间为', run_time)
        yield t + 1, run_time

    with open('./myresult/real_label.txt', 'w', encoding='UTF-8') as f:
        for ct in range(len(real_label)):
            for i in np.array(real_label[ct].cpu()):
                f.write(str(i) + ',')

    with open('./myresult/pre_label.txt', 'w', encoding='UTF-8') as f:
        for ct1 in range(len(pre_label)):
            for i in np.array(pre_label[ct1].cpu()):
                f.write(str(i) + ',')
    # with open('data1_loss.txt', 'w', encoding='UTF-8') as f:
    #     for ls in my_loss:
    #         f.write(str(ls) + ',')
    #
    # with open('data1_ac.txt', 'w', encoding='UTF-8') as f:
    #     for ac in my_ac:
    #         f.write(str(ac) + ',')
    #
    # with open('data1_tac.txt', 'w', encoding='UTF-8') as f:
    #     for tc in tac:
    #         f.write(str(tc) + ',')
    # 保存训练好的模型

    # 读取训练好的模型，加载训练好的参数
    '''
    model = NeuralNetwork()
    model.load_state_dict(torch.load("E:\python_project_pytorch\model\model.pth"))
    '''

    # return (run_time)


if __name__ == '__main__':
    for chunk in train_my_model(5):
        i, time_use = chunk
        print(f'epoch:{i},use_time:{time_use}s')
    print("done!")
