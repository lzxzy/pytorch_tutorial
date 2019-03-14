# PyTorch
## What is Torch.nn Really?
PyTroch提供十分杰出的模块化设计和类如`torch.nn`,`torch.optim`,`Dataset`和`DataLoaer`等来帮助建立和训练神经网络。为了能完全利用它们强大的功能，并针对我们自己的问题定制化，我们需要确切的了解它们到底在干什么。为了达到这一目的，我们先在MNIST数据集上训练一个基础网络，并且不使用上述模块的功能。我们仅仅使用最基础的PtTorch 构建tensor的函数。然后，逐步添加来自`torch.nn`,`torch.optim`,`Dataset`,`DataLoader`的特性，来展示每一部分的功能，以及它们是怎样使得我们的代码更加简洁和灵活的。
### MNIST data setup
我们将使用经典的MNIST数据集，包含黑白手写数组范围1到9。

使用`pathlib`处理路径，通过url请求下载数据。仅在需要的时候引入模块，以便我们准确了解每部分都做了什么。

数据集格式为numpy array格式，使用pickle进行序列化存储。

每张图片大小28×28, 并且存储成一行长度为784（28×28）。

在PyTorch中使用`torch.tensor`而不是numpy array，因此我们需要对数据进行转化。
### Neural net from scratch(no troch.nn)
首先仅通过PyTorch的tensor操作创建一个神经网络。

PyTorch提供随机填充或零填充的tensor方法，我们可以通过这种方法为线性模型初始化权重和偏置。这些都是标准的tensor对象，只需外添加一点：我们需要声明这些tensor需要计算梯度。为了使得PyTorch记录在这些tensor上的所有操作，以便在反向传播过程中可以自动计算梯度。

对于这些权重，我们在进行初始化后设置`requires_grad`，因为我们不想这一步包在内部（使用`_`表明为PyTorch的内建函数）

由于PyTorch可以自动计算梯度，使得我们可以用任何标准Python 函数来建立模型。这里我们使用一系列矩阵乘法和广播加法来建立一个简单线性模型。同时我们需要激活函数，因此我写了一个*log_softmax*函数以供使用。**记住：尽管PyTorch提供了大量预先写好的损失函数，激活函数等等，你依然可以简单的通过标准Python写下自己的实现方法**。PyTorch 甚至可以自动将你的函数快速载入GPU 或向量化CPU代码。

在训练中，每次迭代，我们将执行：
+ 选择一组输入数据（mini-batch）
+ 使用模型进行推断
+ 计算loss
+ `loss.backward()` 更新模型梯度。

接下来使用梯度更新模型参数。在`torch.no_grad()`模块下，因为我们不想这些操作作为下次计算梯度的记录。

然后将梯度归零，为下次迭代做好准备。否则，所有的历史梯度将被累积。
### Using torch.nn.functional
接着将对已有代码进行重构，所做的事情和之前相同，只是采用更高级的`torch.nn`包，使得代码更简洁和灵活。

第一个简化步骤通过`torch.nn.funtional`中的函数取代我们手写的激活和损失函数使得代码更简短（通常为了简便使用名称空间`F`）。这个模块包含了`torch.nn`库中的所有函数（库中的其他部分是各种类）。由于有各种各样的损失和激活函数，同时还有一些方便神经网络搭建的函数，例如池化函数。

如果使用负对数相似损失和对数softmax激活，PyTorch提供了单个函数`F.cross_entropy`将两者组合。因此可以将激活函数从模型中移走。
### Refactor using nn.Module
接着使用`nn.Module`&`nn.Parameter`,为了是训练循环更清楚简洁。使用子类`nn.Module`（自身是一个类可以保持状态追踪）。我们想创建一个类来保有权重，偏置和前向推断的方法。`nn.module`有许多属性和方法（如`.parameters()`和`.zero_grad()`）供我们使用。