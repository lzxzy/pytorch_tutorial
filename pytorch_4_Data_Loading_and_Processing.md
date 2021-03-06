# Pytorch
## Data Loading and Processing 
在机器学习问题中，大量有效的解决问题方法都指向数据预处理。PyTorch提供了许多工具使得数据加载跟加方便和可操作性。
为了使编写的代码更具有可读性。本章将学习如何加载和处理非训练数据集。

## Dataset class
`torch.utils.data.Dataset` 是一个表示数据集的抽象类。我们在PyTorch中实际使用的数据集应该继承`Dataset`并重写下面两个函数：
+ `__len__`,`len(dataset)`返回数据集的大小
+ `__getitem__`为了支持数据集中的数据索引，以便`dataset[i]`可以用来获取 $ ith$ 采样。

接下来我们将创建一个人脸标定数据的数据类。在`__init__`中读取csv数据，并将图片读取放入`__getitem__`中实现内存有效使用，这样可以使得我们不必一次读入全部的图片，只在有需要的时候进行读取。

数据集中每个数据子项是一个字典数据结构`{'image': image, 'landmarks': landmarks}`。同时使用额外的操作`transform`使得我们想对数据项进行额外的处理得以实现。
## Transforms
在上面的数据采样中我们可以发现每个数据样例中图片大小都不同。在实际中大部分神经网络输入的图片大小是固定的。因此， 我们需要编写一些额外的处理代码用来处理图片。本次实验如下：
+ `Rescale`: 处理图片尺寸
+ `RandomCrop`: 对图片随机采样。这是一种数据增广方式。（在一张图片上按照特定尺寸随机裁剪一部分）
+ `ToTensor`: 将numpy格式的图片数据转化为PyTorch中tensor格式（需要交换维度）

我们将编写一个可调用的类而不是一个简单的函数，以免transform的参数在每次调用是都要重复传入。为了实现这个类，只需要实现类中的`__call__`函数，并且如有必要也可以实现`__init__`函数。
## Compose transforms
接下来我们将在一个例子上使用各种变换。
我们需要将图片较短的一遍裁剪到256个像素大小，然后使用一个大小224的方形随机在图片上采样。我们将使用上面实现的类`Rescale`&`RandomCrop`。通过调用`torchvision.transforms.Cmpose`可以使得我们实现这一如下过程。
## Iterating through the dataset
每次数据集采样过程如下：
+ 即时从文件中读取图片
+ 在已读取图片上进行各种变换和预处理
+ 由于变换操作是随机的，数据在被抽样时增广

我们可以简单的使用for循环来加载数据，不过for循环将会丢失一些有用的特性。主要有以下几点：
+ 数据的批特性
+ 数据的随机打乱
+ 并行的加载数据`multiprocessing`

`torch.utils.data.Dataloader`是一个可以提供上述特性的数据迭代器。实验中所使用的参数必须明确。一个比较有趣的参数是`collate_fn`。通过`collate_fn`可以指定批数据采样方式。通常，默认的采样方式已经足够我们使用。

