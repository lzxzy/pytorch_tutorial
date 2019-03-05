# Pytorch
## Torch
torch 是一个类似于Numpy的张量（tensor）操作库，不同的是Torch对GPU有较好的支持，但它的上层包装为lua.

由于lua是一门比较小众的语言，不适合其发展，出现PyTorch。

PyTorch & Torch 使用相同性能的C库： TH，THC，THNN，THCUNN。

结果：PyTorch 和 Torch使用相同的底层，只使用不同的上层接口语言。

## PyTorch
PyTorch 是一个基于Torch的Python开源机器学习库，用于CV，NLP等任务，近来加入RL。By: FIAR
### two feature
- NumPy的替代品，可以使用GPU的强大功能
- 深入学习研究平台，提供最大的灵活性和速度（基于Autograd）

### 对比Tensorflow
pytorch
可读性更强，

## Autograd
在Pytorch中构建神经网络的核心就是autograd包。autograd为Tensor的所有操作提供了自动微分（求导）。define-by-run 框架， 意思是你的backward根据你的代码情况运行，**每个单步循环可以不同**（NLP）

**torch.tensor** 是这个包中的核心类。如果将属性`.requires_grad`设置为`True`，就可以追踪这个tensor对象上的所有操作。
当完成计算后可以通过调用`.backward()`，便可以自动的得到所有梯度。对于该tensor对象的梯度信息将会被加入`.grad`属性。
如果想禁止tensor追踪历史梯度，可以调用`.detach()`来，从历史计算中分离，并且未来的计算也不会被追踪。
为了防止历史追踪（和内存使用），也可以将代码块包裹在 `with torch.no_grad():` 下。

在自动梯度实现中还有另一个重要的类 `Function`
**Tensor** 和 **Function** 相互关联并生成一个非循环图，将完整的计算历史编码。每个tensor对象有一个`.grad_fn`属性依赖于已经创建的Tensor的`Function`（除非这个tensor是用户手动创建的-它的`.grad_fn`是`none`） 。

如果想要计算导数，可以在`Tensor`中调用`.backward()`，如果`tensor`对象是一个标量（只有一个元素），不用给`.backward()`传递参数，如果含有多个元素，需要指定一个`gradient`参数用来表示对应tensor匹配的形状。

通常来讲，`torch.autograd`是一个 vector-Jacobian 积 计算引擎。给定任意一个向量$v = (v_1 v_2 ... v_m)^T$ 计算点积$v^T.J$, 如果$v$恰好是一个标量函数$l=g(\vec{y})$的梯度，也就是说$v=(\frac{\partial{l}}{\partial{y_1}} ... \frac{\partial{l}}{\partial{y_m}} )$, 然后通过链式法则，vector-Jacobian 


