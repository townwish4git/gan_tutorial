# Wasserstein GANs

https://arxiv.org/pdf/1701.07875.pdf

## 问题背景

### GANs的问题

原始GAN中，D和G交替训练。GANs论文中证明了：固定G后将D训练至平衡点，此时再优化G等价于最小化真实数据分布和G生成的伪造数据分布之间的JS散度：

$$\mathop{min}\limits_{G}JS(P_{real}||P_G)$$

但是JS特定情况下存在梯度消失的问题，具体来说：当 $P_{real}$ 和 $P_G$ 的支撑集交集为空时， $JS(P_{real}||P_G)$ 恒等于常数 $log2$ ，这会导致训练过程中梯度为0、训练停滞。

<br>

就实际训练情况而言，当：
1. 生成器G生成的图像效果与真实图像相差较远 $\Rightarrow$  $P_{real}$ 和 $P_G$ 的支撑集交集近似为空

2. 判别器D相较生成器G能更快训练好 $\Rightarrow$ 固定G后将D近似被训练至平衡点 $D^*$

此时优化目标

$$\mathop{min}\limits_{G}\mathbb{E}_ {z\sim Z}[\log (1-D^*(G(z)))]$$

等价于优化两个分布之间的JS散度，且此时JS散度（几乎）恒为常数，因此会出现梯度消失的问题。

<br>

直观上这个问题的出现也是自然的，当判别器训练较好、生成器生成图像质量较差时，D能轻易地判断出图像是否来自真实分布，即此时 $D^*(G(z))$ 近似恒为0， 因此梯度 $\frac{\partial D^*(G(z))}{\partial G(z)}$ 的值很小，出现梯度下降的情况。

<br>

### 一种解决方案

针对上述问题，GANs作者Goodfellow在NIPS 2016给出了一种解决方案。简单而言即是将生成器G训练阶段的优化目标从

$$\begin{equation}\mathop{min}\limits_{G}\mathbb{E}_ {z\sim Z}[\log (1-D^*(G(z)))]\end{equation}$$

修改为

$$\begin{equation}\mathop{max}\limits_{G}\mathbb{E}_ {z\sim Z}[\log (D^*(G(z)))]\end{equation}$$

<br>

直观上看，当判别器训练较好、生成器生成图像质量较差时，生成图片被判别器置信地判为负例，即 $D^*(G(z))$ 近似为0，此时梯度 $\frac{\partial D^*(G(z))}{\partial G(z)}$ 的值很小，也近似为0；但考虑 $\log$ 函数在0附近的导数为正无穷，则 $\frac{\partial\log (D^*(G(z)))}{\partial D^*(G(z))} \approx \infty$ 。由链式法则：

$$\frac{\partial\log (D^*(G(z)))}{\partial G(z)} = \frac{\partial\log (D^*(G(z)))}{\partial D^*(G(z))}\cdot \frac{\partial D^*(G(z))}{\partial G(z)}$$

右式中的第一项近似看作 $\infty$ ，第二项近似看作0，这种 $0\cdot\infty$型的形式“或许”能把整体梯度拉回一个正常的数值范围中。

<br>

数学理论为这种“或许”提供了支撑，简单来说：当把损失函数从（1）改为（2）时，其等价的优化目标从

$$\mathop{min}\limits_{G}JS(P_{real}||P_G)$$

变为了

$$\mathop{min}\limits_{G}[KL(P_{G}||P_{real})-2JS(P_{real}||P_G)]$$

而其中的KL散度在两个分布支撑集交集为空时也不为常数，这保证了生成器效果较差时也不会出现梯度消失的问题。

<br>

### 模式崩溃

但新损失函数的引入又导致了新的问题。一方面其优化目标

$$\mathop{min}\limits_{G}[KL(P_{G}||P_{real})-2JS(P_{real}||P_G)]$$

中的减号意味着KL散度与JS散度彼此冲突。另一方面，KL散度的存在导致了模式崩溃（mode collapse）的问题。


<br>

## WGANs

WGAN中考虑用Earth-Moving/Wasserstein距离替代JS散度:
$EM(P||Q) = \mathop{inf}\limits_{\gamma\sim\prod(P,Q)}\mathbb{E}_{(x,y)\sim\gamma}||x-y||$
，其中:
- $\prod(P,Q)$是所有边缘分布为P和Q的联合分布组成的集合；
- EM/Wasserstein距离：分布P变成分布Q的“最短路径”

Kantorovich-Rubinstein对偶理论可以证明EM距离等价于：
$\begin{equation}EM(P||Q) = \frac{1}{K}\mathop{sup}\limits_{||f||_{L}\leq K}\left( \mathbb{E}[f(P)]- \mathbb{E}[f(Q)] \right)\end{equation}$
当我们因此当我们
  1. 将D限制在L-K连续的范畴中 并且
  2. 迭代训练D使得$\mathbb{E}_{x\sim p_{data}(x)}[D(x)] - \mathbb{E}_{z}[D(G(z))]$最大化
，此时得到的
$D^*\approx\mathop{argsup}\limits_{f:||f||_L\leq K}\left( \mathbb{E}[f(P_{data})] - \mathbb{E}[D(Z)] \right)$
因此此时再训练G使得$\mathop{min}\limits_{G}\left( \mathbb{E}_{x\sim p_{data}(x)}[D^*(x)] - \mathbb{E}_{z}[D^*(G(z))] \right)$近似等价于最小化真实分布和G伪造的数据分布之间的EM距离$EM(P_{real}||P_{g})$ 

注意到
- WGAN的损失函数调整为
  $\mathbb{E}_{x\sim p_{data}(x)}[D(x)] - \mathbb{E}_{z}[D(G(z))]$
  对比GAN中的
  $\mathbb{E}_{x\sim p_{data}(x)}\left[log(D(x)\right] + \mathbb{E}_{z\sim p_{z}(z)}\left[log(1-D(G(z))\right]$
- 我们需要训练D去拟合达到（1）中sup条件的$f$，我们有$f$为K-Lipschiz连续的限制，因此我们设计的鉴别器模型也应当是L-K连续的。为保证这一条件，原始论文中通过将鉴别器模型中的参数截断到$[-c, c]$实现；
-  我们并没有$f:x\mapsto [0, 1]$ 的假定，因此鉴别器D的最后一层无需添加sigmoid层
- 为了使D能够更好拟合满足$\mathop{sup}\limits_{||f||_L\leq K}$ 的$f$，我们会对D训练多轮，因此D和G的训练频次n：1比例往往较大（对比GAN往往1：1地训练D和G）
- 当固定G，多轮训练D后，损失函数近似等价于最小化真实分布和G伪造的数据分布之间的EM距离
  $\mathbb{E}_{x\sim p_{data}(x)}[D(x)] - \mathbb{E}_{z}[D(G(z))]\approx EM\left(P_{data}||G\left(z\right)\right)$
，该损失函数直接量化了伪造数据和真实数据的差异，因此可以被用来作为评价指标来判断生成器模型是否训练充分