import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset

if __name__=='__main__':
    train_ratio = 0.7 # 70%的数据用作训练
    test_ratio= 1 - train_ratio
    # rootdata = r"D:\pack\SEI_v1\PG对讲机样本"
    rootdata = r"F:\研究生\无线课\10800个"
    train_list, test_list = [], []
    data_list = []
    class_flag = -1
    for a, b, c in os.walk(rootdata):
        print(a)
        print(c)
        for i in range(len(c)):
            data_list.append(os.path.join(a,c[i]))
        for i in range(0,int(len(c)*train_ratio)):
            train_data = os.path.join(a, c[i])+ '\t' + str(class_flag)+'\n'
            train_list.append(train_data)
        for i in range(int(len(c) * train_ratio),len(c)):
            test_data = os.path.join(a, c[i]) + '\t' + str(class_flag)+'\n'
            test_list.append(test_data)
        class_flag += 1
    #print(train_list)
    random.shuffle(train_list)
    random.shuffle(test_list)

    with open('train.txt','w',encoding='UTF-8') as f:
        for train_img in train_list:
            f.write(str(train_img))

    with open('test.txt','w',encoding='UTF-8') as f:
        for test_img in test_list:
            f.write(str(test_img))

