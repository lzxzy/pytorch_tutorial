# Pytorch
## Autograd
### PyTorch: Tensor and autograd
通常，手写实现反向传播和求导在小模型中是容易实现的，但在大型的复杂模型中十分困难。

因此，在PyTorch中可以使用自动求导来自动计算导数在神经网络中进行反向传播。PyTorch中的**autograd** 包提供了上述功能。当使用了autgrad，网络计算前向过程时会定义一张计算图。图中的节点就是tenosr对象，边就是用来从输入tensor产生输出tensor的函数。反向传播就是沿着这张图使得你可以方便的计算梯度。

上述过程听起来复杂，在实际使用中很方便。计算图中每个节点代表一个tensor对象。如果`x`是一个tensor，并且属性`x.requires_grad=True`那么`x.grad`就是另一个用来保存`x`的梯度的tensor对象。

### PyTorch: Defining new autograd functions
在底层，每个原始的autograd操做只有两个函数应用在tensor对象上。**forward**对输入tensor计算输出结果tensor。**backward**函数接受输出tensor并计算对应梯度给对应`tensor.gard`。

在PyTorch中，我们可以方便的定义自己的autograd通过定义`torch.autograd.Function`的子类并实现`forward`&`backward`函数。然后我们就可以使用新的autograd操作通过实例化该类并把它当作一个函数进行调用，传入输入tensor。
### TensorFlow: Static Graphs
PyTorch 的autograd与TensorFlow十分相似： 在框架中我们都定义一个计算图，使用自动求导计算梯度。两者之间最大的不同在于TensorFlow的计算图是静态的，而PyTorch使用动态计算图。

在TensorFlow中，我们只定义一次计算图，然后一次又一次的执行相同的计算图，唯一的不同就是输入不同的数据给计算图。在PyTorch中每次forward定义一个新的计算图。

静态图的好处在于你可以在计算之前预先对图进行优化，例如：框架可以确保操作的有效性，或者为在多机或多GPU分布式训练中提出优化策略。如果一遍又一便重复使用统一章计算图，那么潜在的消耗可以通过预先优化分摊到每次计算中。

静态图和动态图的另一个不同在于控制流。对于一些模型，我们可能希望对每个数据点使用不同的计算，例如：使用RNN网络，针对每个数据点可能展开不同数量的神经元，这种展开可以通过循环实现。在静态图中循环结构需要作为图的一部分。由于这一原因，TensorFlow提供了例如`tf.scan`这样的操作将循环映射到图中。动态图在面对种情况就比较简单：由于我们对每次计算即时建立计算图，对于每个不同的输入我们可以使用正常的控制流方式。
## nn module
### PyTroch: nn
计算图和自动梯度在定义复杂操作和自动计算导数时是非常有用的范式，不过对于大型神经网络模型，原生的autograd有些过于底层和低级。

当搭建神经网络时我们经常将计算集中在神经层上（一些具有可学习参数可以通过学习被优化的结构）

在TensorFlow中，类似Keras, TensorFlow-Slim和TFlearn这样的包提供对原生计算图高级的抽象，从而方便我们建立神经网络。

在PyTorch中，`nn`提供类似的功能。`nn`包定义了一系列模型**Modules**，等效于神经网络中的层。一个Module接受一个输入tensor并输出另一个tensor，同时可以保存内部状态例如包含可学习参数的Tensor。`nn`包也定义了一些有用的较为常用的在神经网络中使用的损失函数。

### PyTorch: optim
在此之前我们都是手工更新网络中可学习参数（使用`torch.no_grad()`或者`.datal`来避免历史追踪）。对于简单的优化算法如SGD来说不是什么负担，但在神经网络实际使用过程中我们经常会使用其他优化算法如AdaGrad, RMSProp,Adam 等。

PyTorch中的`optim`包将这些优化算法思想进行抽象并提供常用优化算法的实现。
### PyTorch: Control Flow + Weight Sharing
在动态图的实例中，我们实现一个与以往不同的模型：一个全连接ReLU网络，在每次forward中传送一个在1到4之间随机选择的数字，数字为几就用几次隐藏层，并重复多次使用相同的权重来计算内部隐藏层。

对于上述模型我们可以使用标准的Python控制流来实现循环，并且在定义forward时可以简单的使用同一个模型多次在内部隐层实现参数共享。