# Pytorch
## Training a Classifier
通过学习神经网络搭建，loss计算，和网络中权重的更新。接下来需要讨论：
### What about Data
通常处理的数据有图像，文本，音频，视频等。你可以通过标准的pyhon 包加载数据为numpy array。然后将这些array转换为`torch.Tensor`送入网络作为输入。

特别的，对于视觉任务，pytorch构建了一个包叫`torchvision`，对于一些公开数据集创造了数据加载器，例如Imagenet,CIFAR10,MNIST等，以及对图片数据进行迁移的工具,分别位于`torchvision.datasets`和`torchvision.data.DataLoader`。

本次实验使用CIFAR10 数据集，包含如下几个类‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. CIFAR-10中的图片尺寸为3×32×32,即拥有三个颜色通道高和宽为32×32个像素的图片。

### Training an image classifier
我们将进行如下几个步骤：
+ 通过使用`torchvision` 加载和处理CIFAR10训练和测试数据集。
+ 定义一个卷积神经网络。
+ 定义一个损失函数
+ 在训练数据上训练网络
+ 在测试数据上测试网络效果
 
`torchvision`输出的数据集是PIL图片像素值为[0,1]。对数据进行变换，转换为[-1,1]的tensor对象。

### Define a Convolutional Neural Network
将上个章节中搭建的神经网络作为本章需要训练的神经网络，并将第一层卷积的输入通道改为3对应rgb3通道。
### Define a Loss function and optimizer
使用分类任务Cross-Entropy Loss 和 SGD + momentum 优化器
### Training the network
训练过程中，只需要在一个指定训练轮数（epoch）的for循环中迭代的输入训练数据给网络得到结果，调用`backward`使用优化器即可完成训练。
### Test the network on the test data
 经过上一过程，我们在训练数据集上进行了10轮训练。接着需要检查网络是否学到一些东西。
 
 将通过检查网络预测输出的类别与其对应的真实值来实现。如果预测正确，将其加入正确预测队列。
 
 网络的输出为10个类别的分数，对应类别分数越高，网络就认为图片属于对应的哪一个类别。
### Training on GPU
类似于将Tensor对象推送到GPU器件上一样，你只需要将网络模型也推送到GPU 上即可。然后对应函数将会递归的将模型结构和参数及其对应的Tensor缓存变为CUDA tensor对象。
## OPTIONAL：DATA PARALLELISM
额外部分，我们将学习如何通过使用`DataParallel`进行多GPU加速。
使用GPU 加速模型训练非常简单，只需要把模型推送到GPU上，然后把数据也推送到GPU上即可。

需要注意一点当调用`.to(device)`时，返回的是原始tensor对象的在GPU上的一个新的拷贝而不是重写原始tensor。我们需要给新的tensor在GPU上重新分配空间。

自然也可以在多GPU上执行`forward`&`backward`。不过，在默认情况下pytorch只使用一块GPU。可以通过调用`DataParallel`使得模型并行运行实现多GPU加速

`model = nn.DataParallel(model)`
