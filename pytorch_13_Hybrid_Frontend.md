# Deploying a Seq2Seq Model with the Hybrid Frontend(混合前端)
本章将通过把一个seq2seq模型使用PyTorch的混合前端迁移到Torch脚本的方式来进行讲述。我们将把一个来自Chatbot Tutorial的对话模型进行转换。你可以先学习Chatbot Tutorial部分得到自己的对话模型然后把本章当作上述章节的第二部分进行学习，或者直接使用接下来提供的预训练模型。在学习完本章后，可以参考原始的Chatbot 部分更详细的理解数据处理，模型理论和搭建，以及模型训练等过程。
## What is the Hybrid Frontend
基于深度学习的项目在研究和开发阶段，使用**eager**进行交互是有利的（类似PyTorch的命令交互界面）。这使得用户能够写出熟悉，惯用的PYTHON代码，允许用户使用Python数据结构，控制流操作，打印状态，和单元调试。同时即时交互对研究和实验任务是一个很有用的工具，然而当在生产环境中进行模型开发时，基于图（graph-based）的模型表达方式是十分有效的。一个延迟图表征可以通过优化如无序执行，并且针对硬件架构有针对性进行高效优化。同时，基于图的表征方式可以在框架不可知的情况下也能得到模型输出。PyTorch 提供一种机制用来逐步将即时模式的代码转换成Torch 脚本，Python的一种静态分析和优化子类，Torch独立于python 运行环境表示深度学习任务。

将即时模式的PyTorch程序转换为Torch脚本的API接口包含在`torch.jit`模块中。这个模块包含两个核心的模式：**tracing** & **scripting**。`torch.jit.trace`函数接受一个模型或函数和一组输入样例。然后对样例执行函数或模型同时对计算步骤追踪计数，然后执行跟踪操作输出基于图的函数或模型。**Tracing**对于直接执行的不涉及数据依赖的控制流的模型和函数如，标准CNN网络，是十分有效的。然而，如果一个函数有数据依赖的状态和循环需要被追踪，在样例输入后只有沿着执行路线被调用的操作被记录。换句话说，控制流本身没有被捕获。为了将包含数据依赖控制流的模型和函数转化为Torch脚本，**scripting**机制被提供。**Scripting**显式的将模型转换为Torch脚本，包含所有可能的执行路径。为了使用脚本模式，需要确保从`torch.jit.ScriptModule`（而不是`torch.nn.Module`）中继承基类，同时在Python函数中添加`torch.jit.script`装饰器或给你的模型方法添加`torch.jit.script_method`装饰器。需要注意一点脚本方式仅支持有限的Python子类。

## Prepare Environment
首先，我们将引入一些需要用到的包和常量。如果你计划使用自己的模型，确保`MAX_LENGTH`常量设置正确。提醒一点，这个常量定义在训练中允许的最大的句子长度和模型输出中所能产生的最长的句子。
## Model Overview
如先前提到的，我们将使用`sequence-to-sequence`模型。模型的使用场景为当我们的输入为一个可变长度的序列，输出也为一个可变长度的序列而无序一对一的输入映射。一个seq2seq模型由两个RNN共同工作构成：一个**encoder**和一个**decoder**

### Encoder

