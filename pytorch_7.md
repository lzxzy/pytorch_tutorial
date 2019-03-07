# PyTorch
## Saving and Loading Models
本章提供了各种场景下关于保存和加载PyTorch模型的解决方法。本章的讲解较为轻松，你也可以跳过不感兴趣的部分，只关注你需要的部分。

当谈到保存和加载模型，有三个较为重要的核心函数需要介绍：
+ `torch.save`: 保存一个序列化对象到硬盘。这个函数使用Python中的`pickle`来实现序列化。模型结构，参数tensor，和所有相关对象的字典结构可以通过这个函数进行保存。
+ `torch.load`: 使用`pickle`的反序列化方法将保存的`pickle`对象文件加载到文件中。这个函数也可以用来给设备加载数据。
+ `torch.nn.Module.load_state_dict`： 使用一个反序列化的`state_dict`加载模型的参数字典。

## What is a `state_dict`
在PyTorch中，可学习的参数（如：权重和偏置）在一个`torch.nn.Module`模型中被包含在模型的参数（使用`model.parameters()`获取）。一个*state_dict*是一个简单的Python字典对象将网络中每一层的参数tensor进行映射。要注意的是只有具有可学习参数的层（如：卷积层， 线性层等）在模型中有实例的state_dict。优化器对象`torch.optim`也有state_dict，包含优化器状态信息，和一些使用到的超参数。

由于*state-dict*是一个Python字典，它们可以方便的进行保存，更新，改变和重载，使得PyTorch模型和优化器更具有模块性。
## Saving & Loading Model for Inference
### Save/Load `state_dict`(推荐)
**Save**

`torch.save(model.state_dict(), PAHT)`

**Load**

```
model = TheModleClass()
model.load_state_dict(torch.load(PATH))
model.eval```

当为前向推断保存模型时，只需要保存模型训练过程中已经学到的各种参数。使用`torch.save()`保存模型的*state_dict*将给你在重载模型时最大的灵活性。这也是最推荐的模型保存加载方法。

PyTroch比较方便的是模型保存文件的后缀使用`.pt`&`.pth`都可以。

记住：你必须调用`model.eval()`在推断之前设置**dropout**和**bathch normalization**层为评估模式。如果不这样做，将导致不一致的推理结果。

注意：
> 注意到`load_state_dict()`函数的参数是一个字典对象，而不是模型的保存路径。这意味着你必须在传递给`load_state_dict()`函数之前反序列化已经保存的*state_dict*。例如，你不能直接加载模型通过`model.load_state_dict(PATH)`

### Save/Load Entire Model
**Save**

`torch.save(model, PATH`

**Load**

```
model = torch.load(PAHT)
model.eval()```

上述save/load过程使用更直观的语法并且不涉及太多行代码。使用这种方法保存模型将会调用Python`pickle`模块保存整个模型结构及里面的参数。这种方法的缺点在于保存模型时序列化的数据与特定的类和特定的字典结构绑定。原因在于使用`pickle`保存时并不会同时保存构建模型的类。它同时保存了一个包含类的文件路径，以供在加载模型时使用。由于这个原因，当你在自己工作中使用别人训练好模型，你的代码可能以各种方式终止执行。

## Saving & Loading a General Checkpoint for Inference and/or Resuming Training
**Save**

```
torch.save({'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict()
           'loss': loss,
           ...
           }, PAHT)```
**Load**
```
model = TheModelClass(*args,**kwargs)
optimizer = TheOptimizerClass(*args,**kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint('model_state_dict')
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
//or
model.train()
```
当保存一个通用的检查点，以供推断或继续训练，你不光要保存模型的*state_dict*。保存优化器的*state_dict*也很重要，其中包含有缓存和用来训练更新权重的各种超参数。其他一些你想保存的项目如，中断训练时的轮数，最后记录的训练loss，额外的`torch.nn.Embedding`层，等。

为了保存多个组件，将他们组织为一个字典并使用`torch.save()`将字典序列化。后缀名一般为`.tar`

为了加载这些项，首先初始化模型和优化器，然后使用`torch.load()`加载本地保存的字典数据。然后使用字典方式进行读取。

除了在推断时使用`model.eval()`。如果继续进行训练则需要调用`model.train()`来确保一些层处于训练模式。
## Saving Multiple Models in One File
**Save**

```
torch.save({'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dcit(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            },PATH)
```

**Load**

```
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()
```