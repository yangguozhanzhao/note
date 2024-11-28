# 深度学习公式

* 数学公式和符号推导
    * 常规小写字母表示标量，加粗小写字母表示向量，加粗大写字母表示矩阵和张量。
    * 右下角括号内的数字表示维度。
    * 损失函数和梯度批量全部用单个样本的公式，推广到批量样本都是求均值。
* 有络结构图，没有代码

## 1·线性神经网络模型
### 1.1线性回归模型


n个样本，每个样本m个特征（输入），1个输出，模型表示为：

$$\hat{\mathbf{y}}_{(n,1)} =  \mathbf{X}_{(n,m)}\mathbf{w}_{(m,1)} + b_{(1)}$$

单个样本的损失函数为：

$$l(\mathbf{w}, b) = \frac{1}{2}(\hat{y}_{(1)} - y_{(1)})^2 = \frac{1}{2}(\mathbf{x}_{(1,m)}\mathbf{w}_{(m,1)} + b_{(1)} - \mathbf{y}_{(1)})^2$$

对w和b分别求导为：

$$\frac{\partial_{\mathbf{l}}}{\partial_{\mathbf{w}}}=\frac{\partial_{\hat{y}}}{\partial_{\mathbf{w}}}(\hat{y} - y) = {\mathbf{x}^\top}_{(m,1)}(\hat{y} - y)$$
$$\frac{\partial_{\mathbf{l}}}{\partial_{\mathbf{b}}}=\frac{\partial_{\hat{y}}}{\partial_{\mathbf{b}}}(\hat{y} - y) = \hat{y} - y$$

模型的求解变为寻找一组参数$(\mathbf{w},b)$，使得损失函数$l(\mathbf{w}, b)$ 对所有样本的平均值最小，采用梯度下降法更新参数：

$$ \mathbf{w} \leftarrow \mathbf{w} -\eta \frac{\partial_{l}}{\partial_{\mathbf{w}}} = \mathbf{w}_{(m,1)} - \eta * {\mathbf{x}^\top}_{(m,1)}(\hat{y}_{(1)} - y_{(1)}) \\ b \leftarrow b -\eta \frac{\partial_{l}}{\partial_{\mathbf{b}}} = b_{(1)} - \eta * (\hat{y}_{(1)} - y_{(1)}) $$



n个样本时，损失函数和梯度全部用均值，小批量随机梯度下降法更新，维度不同矩阵相加时广播机制计算（后续模型批量样本时类似）：

$$L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n l_i(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n \frac{1}{2}(\hat{y}_i - y_i)^2 = \frac{1}{n} \sum_{i=1}^n(\frac{1}{2}(\mathbf{x}_{(i,m)}\mathbf{w}_{(m,1)} + b_{(1)} - \mathbf{y}_{(i,1)})^2)$$

小批量随机梯度下降更新参数
$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -\eta \frac{1}{n} \sum_{i=1}^n \frac{\partial_{\mathbf{L}}}{\partial_{\mathbf{w}}} = \mathbf{w}_{(m,1)} - \eta * \frac{1}{n} \sum_{i=1}^n({\mathbf{x}^\top}_{(m,i)}(\mathbf{x}_{(i,m)}\mathbf{w}_{(m,1)} + b_{(1)} - \mathbf{y}_{(i,1)})),\\ b &\leftarrow b_{(1)} -\eta \frac{1}{n} \sum_{i=1}^n \frac{\partial_{\mathbf{L}}}{\partial_{b}} = b_{(1)} - \eta * \frac{1}{n} \sum_{i=1}^n(\mathbf{x}_{(i,m)} \mathbf{w}_{(m,1)} + b_{(1)} - \mathbf{y}_{(i,1)}). \end{aligned}$$ 


### 1.2 softmax回归

处理分类问题，n个样本,每个样本有m个特征，c个输出(总类别数)
$$
\begin{aligned}
\mathbf{O}_{(n,c)} = \mathbf{X}_{(n,m)} \mathbf{W}_{(m,c)} + \mathbf{b}_{(1,c)}, \\
\hat{\mathbf{Y}}_{(n,c)} = \operatorname{softmax}(\mathbf{O}_{(n,c)}) = \frac{\exp(\mathbf{O}_{(n,c)})}{\sum_{j=1}^{c} \exp(\mathbf{O}_{(n,j)})}
\end{aligned}
$$
softmax函数将输出值转换成概率分布（非负且和为1），而且可导，虽然非线性函数，但是输出仍然是输入的仿射函数。还是一个线性模型

单个样本的softmax损失函数为交叉熵损失函数（根据最大似然估计），此处log是自然对数,对y求和就是1：
$$ l(\mathbf{y}_{(1,c)}, \mathbf{\hat{y}_{(1,c)}}) = -\sum_{j=1}^{c} y_{(1,j)} \log \hat{y}_{(1,j)} = -\sum_{j=1}^{c} y_{(1,j)} \log \frac{\exp(o_{(1,j)})}{\sum_{k=1}^{c} \exp(o_{(1,k)})} \\= \sum_{j=1}^{c} y_{(1,j)}\log \sum_{k=1}^{c} \exp(o_{(1,k)})-\sum_{j=1}^{c} y_{(1,j)} o_{(1,j)} =\log \sum_{k=1}^{c} \exp(o_{(1,k)}) - \sum_{j=1}^{c} y_{(1,j)} o_{(1,j)}) $$

求导：
$$ \frac{\partial l_{(1,c)}}{\partial o_{(1,c)}} = \frac{\exp(o_{(1,c)})}{\sum_{k=1}^{c} \exp(o_{(1,k)})} - \mathbf{y}_{(1,c)} = \operatorname{softmax}(\mathbf{o})_{(1,c)} - \mathbf{y}_{(1,c)} $$

梯度就是估计值与真实值之差，与线性回归类似，softmax回归的参数还是线性回归的参数，只是多了一个softmax函数。

为了更新W和b，需要使用链式法则求解L对W和b的偏导（批量n）。

$$  \mathbf{W} \leftarrow \mathbf{W} -\eta * \frac{\partial \mathbf{l}}{\partial \mathbf{W}} = \mathbf{W} -\eta * \frac{\partial \mathbf{l}}{\partial \mathbf{o}} \frac{\partial mathbf{o}}{\partial \mathbf{W}} = \mathbf{W}_{(m,c)} -\eta *\mathbf{x^{\top}}_{(m,1)} (\operatorname{softmax}(\mathbf{o})_{(1,c)} - \mathbf{y}_{(1,c)})   $$

$$ \mathbf{b} \leftarrow \mathbf{b} -\eta * \frac{\partial {\mathbf{l}}}{\partial{\mathbf{b}}} \frac{\partial \mathbf{l}}{\partial \mathbf{b}} = \mathbf{b} -\eta * \frac{\partial \mathbf{l}}{\partial \mathbf{o}} \frac{\partial \mathbf{o}}{\partial \mathbf{b}} = \mathbf{b}_{(1,c)} -\eta * (\operatorname{softmax}(\mathbf{o})_{(1,c)} - \mathbf{y}_{(1,c)})$$


### 1.3 多层感知机MLP

加入隐藏层和非线性激活函数，单个样本，m个特征，隐藏层有h个输出，最后c个输出：
$$ \mathrm{h}_{(1,h)} = \mathrm{x}_{(1,m)}{W^{(1)}_{(m,h)}}+\mathrm{b^{(1)}_{(1,h)}},\\
\mathrm{o}_{(1,c)} = \sigma(\mathrm{h_{(1,h)}}){W^{(2)}_{(h,c)}}+\mathrm{b^{(2)}_{(1,c)}}$$

假设激活函数为ReLU函数，激活函数的导数为：
$$ ReLU(x)=\max(0,x)\\
\frac{\partial \sigma}{\partial x} = \begin{cases} 1 & x>0 \\ 0 & x\leq 0 \end{cases} $$


损失函数为交叉熵损失函数(自带softmax函数)，单个样本损失函数对W和b求导,这里的梯度反向传播需要先更新上层的参数，再根据上层参数更新下层参数：
$$ \frac{\partial l_{(1,c)}}{\partial o_{(1,c)}} =\operatorname{softmax}(\mathbf{o})_{(1,c)} - \mathbf{y}_{(1,c)}  $$
$$ \frac{\partial l_{(1,c)}}{\partial b^{(2)}_{(1,c)}} = \frac{\partial l_{(1,c)}}{\partial o_{(1,c)}} \frac{\partial o_{(1,c)}}{\partial b^{(2)}_{(1,c)}} = \operatorname{softmax}(\mathbf{o})_{(1,c)} - \mathbf{y}_{(1,c)}  $$
$$ \frac{\partial l_{(1,c)}}{\partial W^{(2)}_{(h,c)}} = \frac{\partial l_{(1,c)}}{\partial o_{(1,c)}} \frac{\partial o_{(1,c)}}{\partial W^{(2)}_{(h,c)}} = \sigma(\mathbf{h^\top_{(h,1)}})(\operatorname{softmax}(\mathbf{o})_{(1,c)} - \mathbf{y}_{(1,c)})   $$
$$ \frac{\partial l_{(1,c)}}{\partial b^{(1)}_{(1,h)}} = \frac{\partial l_{(1,c)}}{\partial o_{(1,c)}} \frac{\partial o_{(1,c)}}{\partial h_{(1,h)}} \frac{\partial h_{(1,h)}}{\partial b^{(1)}_{(1,h)}} =  (\operatorname{softmax}(\mathbf{o})_{(1,c)} - \mathbf{y}_{(1,c)})\mathbf{{W^{(2)\top}}_{(c,h)}}  \odot \sigma^\prime(\mathbf{h_{(1,h)}})  $$
$$ \frac{\partial l_{(1,c)}}{\partial W^{(1)}_{(m,h)}} = \frac{\partial l_{(1,c)}}{\partial o_{(1,c)}} \frac{\partial o_{(1,c)}}{\partial h_{(1,h)}} \frac{\partial h_{(1,h)}}{\partial W^{(1)}_{(m,h)}} = \mathbf{{x^\top_{(m,1)}}}[(\operatorname{softmax}(\mathbf{o})_{(1,c)} - \mathbf{y}_{(1,c)})\mathbf{{W^{(2)\top}}_{(c,h)}}  \odot \sigma^\prime(\mathbf{h_{(1,h)}})]$$

### 1.4 权重衰减和暂退法
为了应对过拟合，我们可以尝试使用权重衰减（weight decay）或者暂退法（dropout）的方法。这两种方法都能减小模型的复杂度，使其不至于过拟合。

权重衰减
权重衰减是指在损失函数中添加一个权重衰减项，该项对模型的权重做了一个惩罚。具体来说，给定超参数 $\lambda$，权重衰减项为

$$L_D(\mathbf{w},b) = L(\mathbf{w},b) + \frac{\lambda}{2} \|\mathbf{w}\|^2$$
$$\frac{\partial L_D(\mathbf{w},b)}{\partial \mathbf{w}} = \frac{\partial L(\mathbf{w},b)}{\partial \mathbf{w}} + \lambda \mathbf{w}$$

暂退法
暂退法是指在训练时随机将某些权重置为0，使得网络的某些部分不工作，从而降低模型对特定输入的依赖性。具体来说，给定超参数 $p$，暂退法的损失函数保持不变，但是影响梯度计算，被丢弃的神经元梯度为0。即

$$\frac{\partial L_D(\mathbf{w},b)}{\partial \mathbf{w}} = \frac{\partial L(\mathbf{w},b)}{\partial \mathbf{w}} \odot \mathbf{D}$$

其中 $\mathbf{D}$ 是由 0 和1 组成的随机变量，满足均匀分布。

## 2 卷积神经网络
### 2.1 卷积层

多层感知机适合处理表格数据，但对于高维感知数据，多层感知机可能会变得不实用，因为它需要大量的权重和参数。卷积神经网络（CNN）是一种有效的解决方案。它是由卷积层和池化层组成的网络。

卷积层计算的公式为：
$$ \mathbf{H}_{(1,c_o,w,h)} = \mathbf{W}_{(c_o,c_i,p,q)} \ast \mathbf{X}_{(1,c_i,w+p-1,h+q-1)} + \mathbf{b}_{c_o} $$

$$  h_{c_o,w,h} = \sum_{c_i=1}^{c_i} \sum_{i=0}^{p-1} \sum_{j=0}^{q-1} w_{c_o,c_i,i,j} x_{c_i,i+w,j+h} +b_{c_o} $$

梯度计算的公式为：
$$ \frac{\partial \mathbf{H}_{(1,c_o,w,h)}}{\partial \mathbf{W}_{(c_o,c_i,p,q)}} = \mathbf{X}_{(1,c_i,w+p-1,h+q-1)} $$

$$ \frac{\partial \mathbf{H}_{(1,c_o,w,h)}}{\partial \mathbf{b}_{c_o}} = 1 $$

### 2.2 汇聚层
没有参数，但是影响梯度传播，类似于激活函数，可以提升模型的泛化能力。 

梯度
最大汇聚层：在反向传播时，只有对应于前向传播中最大值位置的梯度被传递回上一层，其他位置的梯度为零
平均汇聚层：在反向传播时，所有位置的梯度均为1/池化窗口大小，即平均池化层的梯度是均匀的。


### 2.3 经典卷积网络的公式表达


## 3 循环神经网络层
### 3.1 循环神经网络层
有隐藏状态的循环神经网络层（RNN）的公式如下：

![](https://p.ipic.vip/7cn1z7.png)

$$\mathbf{h_t}_{(1,h)} = \phi(\mathbf{W_{x}}_{(d,h)} \mathbf{x_t}_{(1,d)} + \mathbf{W_{h}}_{(h,h)} \mathbf{h_{(t-1)}}_{(1,h)} + \mathbf{b_{h}}_{(1,h)}),\\ 
\mathbf{o_t}_{(1,q)} = \mathbf{W_{q}}_{(h,q)} \mathbf{h_t}_{(1,h)} + \mathbf{b_{q}}_{(1,q)}$$


通过时间反向传播（BPTT），梯度计算是整个时间序列T的梯度的平均值,$W_h$梯度很难计算：
$$\ \frac{\partial \mathbf{o_t}_{(1,q)}}{\partial \mathbf{W_{q}}_{(h,q)}} = \frac{1}{T} \sum_{t=1}^T  \mathbf{h_t}_{(1,h)} $$


$$\ \frac{\partial \mathbf{o_t}_{(1,q)}}{\partial \mathbf{W_{x}}_{(d,h)}} = \frac{1}{T} \sum_{t=1}^T \frac{\partial \mathbf{o_t}_{(1,q)}}{\partial \mathbf{h_t}_{(1,h)}} \frac{\partial \mathbf{h_t}_{(1,h)}}{\partial \mathbf{W_{x}}_{(d,h)}} = \frac{1}{T} \sum_{t=1}^T {\mathbf{W_q}_{(h,q)}} {\mathbf{x_t}_{(1,d)}}$$

$$\ \frac{\partial \mathbf{o_t}_{(1,h)}}{\partial \mathbf{W_{h}}_{(h,h)}} = \frac{1}{T} \sum_{t=1}^T \frac{\partial \mathbf{o_t}_{(1,q)}}{\partial \mathbf{h_t}_{(1,h)}} \frac{\partial \mathbf{h_t}_{(1,h)}}{\partial \mathbf{W_{h}}_{(h,h)}} = \frac{1}{T} \sum_{t=1}^T {\mathbf{W_q}_{(h,q)}} ( \mathbf{h_{(t-1)}} + \mathbf{W_{h}}\frac{\partial \mathbf{h_{(t-1)}}}{\partial \mathbf{W_{h}}_{(h,h)}})$$

### 3.2 梯度裁剪
在深度学习中，梯度消失和梯度爆炸是常见的问题。为了应对这一问题，深度学习框架通常提供梯度裁剪（gradient clipping）功能，即对每个参数的梯度进行裁剪，使其永远不会超过某个阈值。具体来说，对于某个参数 $p$，如果它的梯度 $g$ 超过阈值 $\theta$，那么我们就将 $g$ 重新缩放为 $\theta$。这样一来，$p$ 的梯度就不会超过 $\theta$，从而防止梯度爆炸。
### 3.3 多层循环神经网络
![image.png](https://p.ipic.vip/luuhxk.png)
$$\mathbf{h_t}^{l}_{(1,h)} = \phi^l(\mathbf{W_{x}}^{l}_{(d,h)} \mathbf{h_t}^{l-1}_{(1,h)} + \mathbf{W_{h}}^{l}_{(h,h)} \mathbf{h_{(t-1)}}^{l}_{(1,h)} + \mathbf{b_{h}}^{l}_{(1,h)}),\\ 
\mathbf{o_t}_{(1,q)} = \mathbf{W_{q}}_{(h,q)} \mathbf{h_t}^{l}_{(1,h)} + \mathbf{b_{q}}_{(1,q)}$$

### 3.4 GRU
![gru](https://p.ipic.vip/ibuxli.png)

$$ \mathbf{R}_t = \sigma(\mathbf{X}_t\mathbf{W}_{xr}  + \mathbf{H}_{t-1}\mathbf{W}_{hr}  + \mathbf{b}_r)$$
$$ \mathbf{Z}_t = \sigma(\mathbf{X}_t\mathbf{W}_{xz}  + \mathbf{H}_{t-1}\mathbf{W}_{hz}  + \mathbf{b}_z)$$
$$ \mathbf{\hat{H}}_t = tanh(\mathbf{X}_t\mathbf{W}_{xh}  + \mathbf{R}_t \odot  \mathbf{H}_{t-1}+ \mathbf{b}_h$$
$$ \mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1} + (1 - \mathbf{Z}_t) \odot \mathbf{\hat{H}}_t$$

### 3.5 Seq2Seq模型

![d2l-images.png](https://p.ipic.vip/r3jfpy.png)

编码器：
$$ \mathbf{h_t}= \text{f}(\mathbf{x}_t, \mathbf{h}_{t-1}) $$
$$ \mathbf{c}= \text{q}(\mathbf{h_1},\ldots,\mathbf{h_T}),如 \mathbf{c} = \mathbf{h_T} $$

解码器：
$$ \mathbf{s_{t^\prime}} = \text{g}(cat(\mathbf{y}_{t^\prime-1}, \mathbf{c}), \mathbf{s}_{t^\prime-1}) $$

## 4 注意力机制
### 4.1 注意力机制公式
缩放点积注意力公式（Scaled Dot-Product Attention），查询和建有相同的长度d：
$$\text{Attention}(Q, K, V)=\text{softmax}(\frac{Q_{(n,d)}K^T_{(d,m)}}{\sqrt{d}})V_{(m,v)}$$

### 4.2 Bahdanau(seq2seq-attention)注意力

![image.png](https://p.ipic.vip/blow9n.png)

相比seq2seq变化点：

1. 编码器在所有时间步的输出，将作为注意力的key和value; 
2. 解码器RNN对上一个词的输出是query; 
3. 注意力的输出和下一个词的词嵌入合并作为解码器RNN的输入。

增加注意力编码器：
$$ \mathbf{h_t}= \text{f}(\mathbf{x}_t, \mathbf{h}_{t-1}) $$
$$ \mathbf{c_{t^\prime}}= \sum_{t=1}^T \alpha (\mathbf{h_t}, \mathbf{s_{t'-1}}) \mathbf{h_t}  $$

解码器：
$$ \mathbf{s_{t^\prime}} = \text{g}(cat(\mathbf{y}_{t^\prime-1}, \mathbf{c_{t^\prime}}), \mathbf{s}_{t^\prime-1}) $$

### 4.3 多头自注意力机制
![image.png](https://p.ipic.vip/m6wr3d.png)
自注意力公式，k，q，v都是同一组输入，可以并行计算
$$ \mathbf{y_i} = f(\mathbf{x_i} , (\mathbf{x_1},\mathbf{x_1}),\dotsc,(\mathbf{x_n},\mathbf{x_n}))  $$


### 4.4 transforms模型

![image.png](https://p.ipic.vip/d3hkfc.png)