import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as scio
# 定义一个函数，要注意label_pred和label_true都必须是np.array()

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(  # 这是我们要学习的bincount函数
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    # minlength属性规定了bincount函数返回的数组的最小长度，用0补齐
    # print(hist)
    return hist


def result_train(path=r"D:\user\cuirui\chenpeng\signal\sei_data"):
    #输出读取到的数据
    real_label_data = pd.read_csv("real_label.txt",header=None)
    real_label_data = np.array(real_label_data.values[0][0:-1])
    pre_label_data = pd.read_csv("pre_label.txt",header=None)
    pre_label_data = np.array(pre_label_data.values[0][0:-1])
    new_real = np.zeros((len(pre_label_data)),dtype=int)
    for i in range(len(real_label_data)-1):
        new_real[i] = int(real_label_data[i])

    binc = np.bincount(new_real)
    new_pre = np.zeros((len(pre_label_data)),dtype=int)
    for i in range(len(pre_label_data)-1):
        new_pre[i] = int(pre_label_data[i])
    #通过下面这个循环，可以计算出我们的分类情况
    prediction = new_pre
    truth = new_real
    hist = np.zeros((len(binc),len(binc)))
    for lp,lt in zip(prediction,truth):
        hist += _fast_hist(lp.flatten(),lt.flatten(),len(binc))
        print(hist)
        print('\n\n')
    scio.savemat('conf_mat.mat',{"cfm":hist})
    iu = np.diag(hist)
    print(iu)
    print(binc)
    result = iu/binc
    result = np.around(
        result,  # numpy数组或列表
        decimals=3  # 保留几位小数
    )
    print(result*100)
    result = result*100
    plt.figure()
    #plt.show()
    label_list = os.listdir(path)
    # ---------
    data = np.array(result)
    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    for i in range(len(data)):
        plt.bar(label_list[i], data[i])
    plt.title("测试结果分析")
    plt.xlabel("分类")
    plt.ylabel("正确率")
    plt.xticks(label_list,rotation=335,fontsize=7)
    for i, j in zip(label_list, data):
        plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=10)
    # plt.show()
    plt.savefig("test.png")
    plt.close()

if __name__=='__main__':
    result_train(r"F:\研究生\无线课\学长给的\原数据")