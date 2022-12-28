import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import scipy.io as scio

# 数据归一化与标准化
class LoadData(Dataset):
    def __init__(self, txt_path):
        self.imgs_info = self.get_images(txt_path)

        # self.transform = transforms.Compose([
        #     #transforms.Resize(224)
        #     # transforms.RandomHorizontalFlip(),
        #     # transforms.RandomVerticalFlip(),
        #     # transforms.ToTensor(),
        #     # transform_BZ
        # ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        return imgs_info

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        # img = scio.loadmat(img_path)
        # img = Image.open(img_path)
        # img = img.convert('F')
        label = int(label)
        ###############
        with open(img_path) as file_ob:  # 读txt文件
            contents = file_ob.read().split("\n")
        read_data = np.array(contents[:-1], dtype=np.float32)
        read_data = read_data / np.max(read_data)  # 归一化
        # 标准化# 注意归一化和标准化需将I路和Q路数据一起
        ######## 如果分别对I路、Q路做这些线性变换，相当于人为引入了IQ unbalance,不利于分析
        read_data = torch.tensor(read_data)
        img_mean = torch.mean(read_data)
        img_std = torch.std(read_data)
        input_img = (read_data - img_mean) / img_std
        #################IQ分路，组合成2*n的矩阵###########################
        I = input_img[::5]
        Q = input_img[1::5]
        power=input_img[2::5]
        rect_spec_I=input_img[3::5]
        rect_spec_Q=input_img[4::5]


        I = I[0:1024]
        Q = Q[0:1024]
        power=power[0:1024]
        rect_spec_I=rect_spec_I[0:1024]
        rect_spec_Q=rect_spec_Q[0:1024]

        I = I.unsqueeze(0)
        Q = Q.unsqueeze(0)
        power=power.unsqueeze(0)
        rect_spec_I=rect_spec_I.unsqueeze(0)
        rect_spec_Q=rect_spec_Q.unsqueeze(0)
        #st = I + 1j*Q
        #abs_iq = abs(st)
        #img_dct = cv2.dct(np.array(abs_iq))
        # print(I.shape)

        # IQ = torch.cat([I, Q], dim=0)
        #IQ = torch.cat([power], dim=0)
        #IQ = torch.cat([rect_spec_I,rect_spec_Q], dim=0)
        IQ = torch.cat([I, Q, power, rect_spec_I,rect_spec_Q], dim=0)
        # IQ = torch.cat([I, Q, abs_iq, torch.tensor(img_dct)], dim=0)
        ###############

        return IQ, label

    def __len__(self):
        return len(self.imgs_info)


if __name__ == "__main__":
    train_dataset = LoadData("train.txt")
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(image)
        # img = transform_BZ(image)
        # print(img)
        print(label)

    # test_dataset = Data_Loader("test.txt", False)
    # print("数据个数：", len(test_dataset))
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=10,
    #                                            shuffle=True)
    # for image, label in test_loader:
    #     print(image.shape)
    #     print(label)
