# Pytorch 模型并行
模型并行广泛使用于分布式训练技巧上。在Pytorch中并行有两种方式，一种是DataParallel，数据并行，另一种是就是ModelParallel模型并行。

数据并行主要将完整的模型拷贝到多个GPU上，每个GPU读入输入数据的不同部分，通过增加数据输入量，提高了模型效率。而模型并行主要用于单GPU无法满足运行条件的大型模型进行训练而使用，相比于数据并行，模型并行将单个模型分割到不同的GPU上，而不是把一个完整的模型拷贝到不同GPU上。（例如：一个10层的模型，使用数据并行，每个GPU都保存了完整的10模型，而模型并行会将模型分割为2个5层模型分别放入两个GPU中。）

一种比较高级的模型并行思想是将一个大型模型的不同子模型放入不同的GPU中，并通过将中间输出在不同器件上移动实现前向过程。本次实验不会尝试构建一个大型模型并将他们放入有限数量的GPU中。而是专注于展示模型并行的思想。

## 基础用法

让我们先从一个简单的由两层线性层构成的模型开始。为了使模型能在两个GPU上运行，只需要简单的将每个线性层放入不同的GPU即可，然后将输入和中间输出移动的对应层所在的器件上即可。
```python
import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))
```

## 在已有模型上应用模型并行

将一个单GPU模型改为多GPU并行模型也是可行的，只需要修改很少的部分。下面的代码展示了如何分解`torchvision.models.resnet50()`到两个GPU上。实验思想是先继承已有的`ResNet`模型，并在构建模型过程中分离模型的层到不同的GPU上。然后，重写`forward()`方法，通过移动中间输出结果将两个子模型连接起来保持一致。

```python
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))
```

上述实验方法当模型过大无法在单GPU上进行训练的问题。但是，你可能已经注意到相比于运行在单GPU上，多GPU训练模型要慢一些。这是由于，在任何时刻两块GPU中仅仅只有一块在工作，而另一块这保持不工作的状态。而由于需要将中间输出从`cuda:0`复制到`cuda1`上则进一步加剧了速度变慢的情况。

下面的实验将通过量化的方式查看执行时间。在这个实验中，我们通过输入随机生成的数据的对应标签分别训练`ModelParallelResNet50`和`torchvison.models.resnet50()`。经过训练的模型并不能产生任何有用的预测结果，但是我们可以获得可信的运行时间。

```python
import torchvision.models as models

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()
```

上面的`train(model)`方法使用`nn.MSELoss`作为损失函数，`optim.SGD`作为优化器。模拟训练大小为128x128的分为3个batch每个batch120的图片数据。然后，我们使用`timeit`来执行`train(model)`方法10次并绘制执行时间的标准差图像。

```Python
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import timeit

num_repeat = 10

stmt = "train(model)"

setup = "model = ModelParallelResNet50()"
# globals arg is only available in Python 3. In Python 2, use the following
# import __builtin__
# __builtin__.__dict__.update(locals())
mp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

setup = "import torchvision.models as models;" + \
        "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
rn_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)


def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


plot([mp_mean, rn_mean],
     [mp_std, rn_std],
     ['Model Parallel', 'Single GPU'],
     'mp_vs_rn.png')
```

结果表明模型并行的执行时间比单GPU模型实现多7%。因此我们可以得出结论GPU之间tensor的前向和反向拷贝大约有7%的开销。这之中有很大的提升空间，因为我们之前已经发现在执行过程中两块中总有一块处于不工作状态。一种操作方式是进一步分离每个batch为通道，例如：当分离出的一部分运行至第二块卡上时，接下里被分离的部分就被送入第一块卡中。通过这种方式，两个连续的被分离的部分可以同时在两块卡上被计算。
## 使用管道输入进行加速
下面的实验，我们进一步将120张图片的batch分为20张图片一组。由于PyTorch 异步的运行CUDA操作，实现方式上不需要实现多线程并发操作。
```python
class PipelineParallelResNet50(ModelParallelResNet50):
	def __init__(self, split_size=20, *args, **kwargs):
    	super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size
        
    def forward(self, x):
    
    	splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
    	ret = []
        
        for s_next in splits:
        	s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1))
            s_prev = self.seq1(s_next).to('cuda:1')
    	s_prev = self.seq2(s_prev)
    	ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
    return torch.cat(ret)
    
setup = "model = PipelineParallelResNet50()"
pp_run_times = timeit.repeat(
	stmt, setup, number=1, repeat=num_repeat, globals=globals())
pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

plot([mp_mean, rn_mean, pp_mean],
     [mp_std, rn_std, pp_std],
     ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
     'mp_vs_rn_vs_pp.png')
```
注意一点，GPU间的tensor拷贝工作是异步的。如果你创建多个工作流，需要确保拷贝操作以正确的异步方式进行。在未完成拷贝操作时修改原始tensor或者读写目标tensor会导致未知的错误。先前的实现仅在源和目标器件上使用了默认的流，因此不需要强制增加异步操作。

实验结果表明，使用管道输入的方式，并行化的ResNet50加速了`3.75/2.51-1=49%`。距离100%的加速还有一定距离。由于我们在实现过程中已经有了新的参数`split_size`，它对模型运行速度的影响目前并不清楚。直观来讲，使用小的`split_sizes`导致许多小的CUDA核运行，同时使用大的`split_size`结果则是有相关的kernal长时间的等待。因此有可能通过优化`split_sizes`来进一步提升效率。
```python
means = []
stds = []
split_sizes = [1, 3, 5, 8, 10, 12, 20, 40, 60]

for split_size in split_sizes:
	setup = "model  = PipelineParallelResNet50(split_size=%d)" % split_size
    pp_run_time = timeit.repeat(
    	stmt, setup, number=1, repeat=num_repeat, globals=globals())
    means.append(np.mean(pp_run_times))
    stds.append(np.std(pp_run_times))

fig, ax = plt.subplots()
ax.plot(split_sizes, means)
ax.errorbar(split_sizes, means, yerr=stds, ecolor='red', fmt='ro')
ax.set_ylabel('ResNet50 Execution Time (Second)')
ax.set_xlabel('Pipeline Split Size')
ax.set_xticks(split_sizes)
ax.yaxis.grid(True)
plt.tight_layout()
plt.savefig("split_size_tradeoff.png")
plt.close(fig)
```
结果表明设置`split_size`为12可以达到最快的速度，加速了`3.75/2.43-1=54%`。
