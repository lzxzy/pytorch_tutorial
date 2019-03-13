# PyTorch
## Finetuning Torchvision Models
本章将深入了解如何对`torchvision model`进行调优和特征抽取。我们将从内部深入了解几个经典CNN结构如何工作，并建立微调任何PyTorch模型的直觉感受。由于各种模型结构不同，没有以一个统一的调优代码模板可以在所有场景下工作。但是，研究人员必须针对已经存在的结构建立适合模型的结构。

实验中将展现两种迁移学习：微调和特征抽取。在微调中，我们使用一个预训练模型并针对新的任务更新模型的所有参数，本质上是训练新的模型。在特征抽取中，我们使用预训练模型并只更新最后一层用来直接输出预测的权重。将只称为特征抽取是由于我们使用预训练CNN作为固定的特征抽取器，只改变输出层。

通常来说，上述迁移学习方法遵循下面相似的步骤：
+ 初始化预训练模型
+ 针对新的数据集将最后一层的输出个数与输出类别匹配
+ 选择优化算法，以便训练中更新我们想学习的参数
+ 执行训练步骤

### Inputs
使用`hymenpotera`数据集，包含**bees**&**ants**两类数据。设置数据目录`data_dir`。`model_name`是模型结构名字。

其他的输入如下：`num_classes` 数据集中类个数，`batch_size`批大小（和机器性能相匹配）,`num_epochs`我们想训练的轮数，`featyre_extract`布尔值用来决定对模型调优还是特征抽取。
### Model Training and Validation Code
`train_model`函数将给定的模型用来训练和验证。它的输入有，一个给定的模型，一个数据加载字典，一个损失函数，指定的训练轮数，一个布尔变量决定模型调优还是特征抽取。`is_inception`参数是用来适应incetpion v3模型，由于该模型结构使用一个辅助输出。该函数训练指定轮数模型并在每个epoch后进行一次验证。同时对表现最好的模型进行追踪（取决于验证准确率），并在训练最后返回最好的模型。每次训练打印训练和验证准确率。
### Set Model Parameters'.requires_grad attribute
这个函数将模型中的参数`.requires_grad`参数设置为False,当我们进行特征抽取时。默认情况下，当我们加载一个预训练模型，所有模型参数属性都为`.requires_grad=True`，方便我们正常训练或微调。不过当我们在进行特征抽取时我们只希望对新初始化的层做梯度计算，然后不对其他参数计算梯度。
### Initialize and Reshape the Networks
接下来进入最有趣的部分。下面我们对每个网络进行裁剪。需要注意的是我们无法实现自动化而是对不同网络分别进行操作。对于一个CNN模型，它的最后一层通常为一个全连接FC层，有与输出类别数目相同的节点。由于所有模型都在Imagenet上进行过预训练，它们的输出层大小都为1000。这里我们的目标是针对输入的类别将最后一个输出层大小裁剪为合适的数目。

当在特征抽取中， 我们只想更新最后一层的权重，换句话说，我们只想对裁剪过的层计算梯度。因此，我们没有必要对未进行更改的参数计算梯度，所以为了方便我们将`.requires_grad`属性设置为False。这点很重要因为该属性默认为True。然后，当我们初始化一个新的层，默认的新参数属性`.requires_grad=True`,所以只有新的层的参数将被更新。

### Load Data
既然我们知道输入大小是固定的，我们可以初始化数据通过data transforms, image datasets, dataloaders.
### Create the Optimizer
既然模型结构确立，接下来的步骤就是创建优化器用来更新特定的参数。在加载预训练模型后，对模型裁剪前，如果`feature_extract=True`我们手工将参数`.requires_grad`设为False。然后，重新初始化层的参数默认`.requires_grad=True`。所以所有参数`.requires_grad=True`需要被优化。我们列出这些参数并传递给SGD。
### Run Training and Validation Step
最后一步计算loss，运行训练和验证函数。
### Comparison with Model Trained from Scratch
