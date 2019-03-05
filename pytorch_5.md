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
