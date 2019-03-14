# Pytorch

## neural networks
神经网络通过`torch.nn`包进行构建。
通过对`autograd`的快速了解，`nn`依赖`autograd`来定义和区分模型。`nn.Module`包含不同的神经网络层，`forward(input)`返回输出结果。

一个典型的神经网络训练过程包含一下几个步骤：
+ 定义网络结构和一些需要学习的参数（weights）
+ 将数据集重复当作输入
+ 通过网络对输入进行处理
+ 计算loss(网络输出值与真实值的差异)
+ 将梯度反向传播给网络中的参数
+ 更新网络中的参数，通常用一个简单的更新法则：$weight = weight - learning_rate*gradient$

你只需要定义`forward()`函数，`backward()`函数（用来计算梯度）通过使用`autograd` 自动进行定义。你可以在`forward`过程中使用任何Tensor操作。

模型中可学习参数通过`net.parameters()`返回。

随机定义一个32×32 tensor 作为输入（LeNet的输入尺寸），通过`forward`得到输出，将网络中参数的梯度缓存置零，调用`backward`函数进行反向传播。

**注意**
***
`torch.nn` 仅支持mini-batch作为输入。整个`torch.nn`包仅支持输入是数据的mini-batch采样而不是单张图片。

例如：`nn.Conv2d` 接受一个4d Tensor（`batch_size×channels×height×width`）作为输入，如果将单张图片作为输入，需要将输入resize维度0作为batch_size 1。
***
## Loss Function
一个loss function接受一对（输出，对应gt）作为输入，并且计算得到一个值评估计算值与真实值的差距。

当得到loss值后，可以调用`loss.backward()`，计算整张图相对于`loss`的导数，同时图中所有属性`requires_grad=True`的Tensor都有其对应的累积梯度 `.grad` Tensor。

## Backprop
为了将误差进行反向传播，所有我们所需要做的就是调用`loss.backward()`. 同时需要清楚已经存在的梯度，否则当前梯度将累积在已有的梯度上面。

## Update the weights
实际中最简单实用的参数更新方式是**随机梯度下降法(SGD stochastic gradient descent)**
$$weight = weight - learning_rate * gradient$$
你可以手动实现上述过程，然而在实际使用神经网络时，通过希望可以使用不同的优化策略 SGD, Nesterov-SGD, Adam等，为了方便使用，可以调用`torch.optim` 已经实现了各种常用优化器。


