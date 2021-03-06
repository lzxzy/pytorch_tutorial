# Adversarial Example Generation
阅读本章前，希望你可以对一些模型的有效性可以有所理解。在研究过程中一般在模型速度，进度和有效性上持续推进。可是，最近一些观点认为设计和训练模型的安全性和鲁棒性，尤其在类似人脸识别这种场景中对抗想欺骗模型的人。
本章将使你意识到模型的安全漏洞，并给出最近较火的对抗机器学习主题内部视角。也许你会惊奇的发现对图片添加一些难以察觉的绕动，模型的表现效果会完全不同。本章我们将以图片分类中的例子来探索这个主题。我们将使用第一个也是最流行的一个攻击方法，快速梯度标记攻击（FGSM），来欺骗MINIST分类器。
## Threat Model
通常，存在有许多种类的对抗攻击，每种攻击方式有不同的目标以及对攻击理论有不同假设。不过，通常来说总体的目标是给输入数据添加最少的扰动来造成模型所期望的错误分类。一般有好几种对攻击理论的假设，其中两个是：**白盒子**和**黑盒子**。白盒攻击假设攻击者对于模型有完全的理解并且和模型有交互，包括模型结构，输入，输出和模型权重。黑盒攻击则假设攻击者仅与模型输入和输出有交互，并且对模型内部结构和权重完全没有了解。同时对于攻击目标也有不同类型，包含**误分类（misclassification）**和**源/目标 误分类(goal/target misclassification)**。误分类意味着对抗样本仅仅想使模型输出的分类信息是错的但对于新输出的类别并不关心。源/目标误分类的目标则是攻击想改变一张图片并使模型输出特定的错误分类类别。

在本章中，FGSM攻击是一个白盒攻击目标为误分类。有了上述背景知识，我们现在可以开始讨论攻击细节。

## Fast Gradient Sign Attack
在对抗攻击中较早并较出名的一个方法叫做*Fast Gradient Sign Attack(FGSM)*快速梯度标记攻击，由Goodfellow所提出。这种攻击十分有效并且很直观。它的设计通过利用神经网络学习的方式梯度进行攻击。思想很简单，攻击者通过调整输入数据基于相同的反向梯度传播最大化损失LOSS，而不是基于反向梯度传播调整权重来最小化损失loss。换句话说，攻击者使用loss梯度修改输入数据，并调节输入输入数据至损失loss最大化。

在看实现代码之前，让我们现看一下最著名的FGSM的熊猫例子并从中提取一些信息。

从图片中$x$是原始输入图片正确分类为“panda”，$y$是$x$的真实标签，$\theta$代表模型参数，$j(\theta,x,y)$是用来训练网络的loss。攻击模型反向传播梯度给输入图片来计算$\nabla J(\theta,x,y)$。然后，调整输入数据用很小的步长（$\varepsilon$或者图中0.007）在梯度方向（例如 $sign(\nabla_xJ(\theta,x,y))$）使得loss最大化。结果的干扰图片$x'$被目标网络误分类为“gibbon” 当它明显看起来是只熊猫时。

## Implementation
在这部分，我们将讨论输入参数，定义攻击模型，然后运行相关测试。
### Inputs
这里仅有三个输入，定义如下：
+ **epsilons** 一组运行使用的epsilon值。在一组值中有0值是非常重要的因为它表示模型最初始的测试结果。同时，直观上我们期望扰动值越大，会有更多可用的扰动而不是在降低模型精度方面更有效的攻击。由于数据的范围为[0,1]，epsilon的值不应该超过1。
+ **pretrained_model** 在MNIST数据集上预训练的模型。
+ **use_data** 布尔标志来决定是否使用CUDA。注意，使用CUDA的GPU在这篇教程中不是特别重要的CPU也不会消耗太多时间。

### Model Under Attack
如之前提到的, 攻击模型是相同的MNIST模型。你可以训练自己的MNIST模型或者下载已经训练好的模型。网络定义和测试数据加载器是从MNIST例子中复制的。这部分的目标是定义网络和数据加载器，然后初始化模型并加载预训练权重。
### FGSM Attack
现在，我们可以定义函数，通过对原始图片加入扰动构建对抗样本。`fgsm_attack`函数接受三个输入，*image*是原始的干净图图片$x$，*epsilon*是像素级扰动量级$\epsilon$，*data_grad*是输入图片计算得到的梯度$\nabla_xJ(\theta, x, y)$。然后函数按照如下方式创建扰动图片：
$$perturbed\_image = image + epsilon * sign(data_grad) = x + \epsilon * sign(\nabla_xJ(\theta,x,y))$$
最后，为了控制数据的原始范围，扰动图片被裁剪到[0,1]之间。
### Testing Function
最后，这篇教程的核心结果来自`test`函数。每一次对test函数的调用在MNIST测试数据集上执行一个完整的测试步骤并得到最后的准确率。同时，要注意到这个函数也接受epsilon作为输入。这是因为`test`函数得到的准确率是在攻击模型下使用$\epsilon$对抗攻击得到的。更进一步，对于每一个测试集中样例，函数计算输入数据的解析梯度，使用`fgsm_attack`创建干扰图片，然后检查扰动样本是否具有对抗能力。然后测试模型精度，函数同时保存并返回一些成功的对抗样本以供可视化。
### Run Attack
最后的实现部分就是让攻击模型运行起来。这里，我们对每一个epsilon值运行一个完整的测试。对于每一个epsilon值我们同时保存最后的准确率和一些成功的对抗样本用来最后展示出来。要注意到在准确率降低的同时epsilon在增大。注意$\epsilon=0$的条件代表没有进行攻击的原始测试准确率。
## Results
### Accuracy vs Epsilon
第一个结果是准确率对应epsilon的图。如前所述，由于epsilon增加我们期望准确率会降低。这是由于更大的epsilons意味着我们采取了更大的步长在使loss最大化的方向上。注意到曲线变化并不是线性的即使epsilon值在线性空间内变化。例如，准确率在$\epsilon=0.05$时相对于$\epsilon=0$仅降低4%，但准确率在$\epsilon=0.2$时相对于$\epsilon=0.15$时降低25%。同时，注意到模型的准确率在10个分类类别上达到随机准确率在$\epsilon=0.25$到$\epsilon=0.3$之间。