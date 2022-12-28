Create_dataset.py 是用来创建数据集的
操作步骤：
①需要把类别1、类别2。。。分别放到所属类别的文件夹里边，有10个类就建10个文件夹，文件夹名字01，02，03，04.。。。。
②把这些文件夹放到同一个父文件夹里边，注意这个父文件夹下面只能有这些类别文件夹
③把代码里的路径改成父文件夹，然后按自己的需求调整一下训练集和测试集的比例，我这里设置的是7:3
④运行过后就会生成train.txt和test.txt


Create_dataLoader.py 是用来对数据进行预处理的，同时可以打包数据成(data,label)对

model_ex.py里面放了很多模型，有ALexNet\VGG16\RESNET18\RESNET50\RESNET34还有个reception，现在代码里默认调用的是resnet34

train_my_torch_model.py就是用来训练和测试模型的，他调用前面三个文件里的数据、模型、函数，训练完成后会输出pre_label.txt和real_label.txt，有了这两个文件，再运行see_confmatrix.py就能得到混淆矩阵了，会生成一个图片和.mat文件，.mat里存了混淆矩阵。