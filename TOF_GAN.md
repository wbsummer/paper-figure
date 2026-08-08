# 实验 11：DeepLabV3++GAN + RGST-TOFAA

本文汇总实验 11 的实际网络、模型规模与训练损失搭建，供后续绘制框架图使用。

## 1. 网络组成与模型规模

| 网络 | 作用 | 参数量 | FP32 参数存储 |
|---|---|---:|---:|
| 生成器 `DeepLabV3Plus + RGST-TOFAA` | Depth/IR → 5 类材质 logits | **41,094,343** | **156.762 MiB** |
| 判别器 `TOFPhysicsDiscriminator` | 判别分割图是否符合当前 Depth/IR 观测 | **803,459** | **3.065 MiB** |
| 合计 | 训练期 G + D | **41,897,802** | **159.827 MiB** |

参数存储不包含梯度、AdamW 优化器状态和激活图。推理期只保留生成器。

```text
Depth + IR ─► Generator (DeepLabV3+ + RGST-TOFAA) ─► 5-class logits
                  │                                      │
                  │                                      └─► straight-through one-hot segmentation
                  │                                                        │
                  └────────────────────────────────────────────────────────┼─► Conditional Discriminator
Depth + IR ─────────────────────────────────────────────────────────────────┘
```

生成器的 RGST-TOFAA 位于 ASPP 后的 `18 x 20` 高层语义尺度；原始 `layer1` 低层边界特征直接进入 Decoder，不经过 TOFAA 重写。判别器同时使用 5 通道分割图、Depth 与 IR，输出 `71 x 79` 的 Patch 评分图及物理不一致图。

## 2. 训练批次与时序

每个训练批次为 `B=8` 对齐的 `(Depth, IR, mask)`，空间尺寸为 `576 x 640`。训练集实际启用同步水平翻转（概率 `0.5`）、同步旋转（概率 `0.3`、角度 `±10°`）和观测驱动传感器噪声；验证与推理不应用这些增强。

每个 epoch 内的有效顺序为：

1. epoch 0–49：只更新 G，使用分割损失；D 不进行更新。
2. epoch 50 起：先更新 D 一次，再更新 G 一次（`d_steps_per_g=1`）。
3. G 更新期间，D 切换到 eval 并冻结全部参数；D 仅提供对 G 的可导评分。
4. 所有更新使用 AMP 和 L2 梯度裁剪，阈值为 `2.0`。

## 3. 生成器分割损失

设生成器输出 logits 为 `z`，标注为 `y`。分割损失为：

\[
\mathcal{L}_{seg} = 1.0\mathcal{L}_{Dice} + 0.5\mathcal{L}_{Focal} + 1.0\mathcal{L}_{CE}^{LS}
\]

| 项 | 实际配置 |
|---|---|
| Dice | 权重 `1.0` |
| Focal | 权重 `0.5`，`gamma=2.0`；配置中的 `alpha=0.25` 仅对二分类生效，因此对当前 5 类 Focal 实际不参与加权 |
| Cross Entropy | 权重 `1.0`，标签平滑 `0.05` |
| 类别权重 | 未启用（`null`） |
| L1 / Boundary / Lovász | 未启用，权重均为 `0` |

Dice 对 5 个类别分别在有效像素上计算 softmax Dice 后平均；忽略索引为 `255`。标签平滑 CE 对真实类别分配 `0.95` 概率，对其余 4 类各分配 `0.0125` 概率。Focal 以真实类别的 `p_t` 计算 `(1-p_t)^2[-\log(p_t)]`，再乘总权重 `0.5`。

## 4. GAN 逐步启用策略

基础 GAN 权重为 `lambda_gan=0.02`。当前 epoch 为 `e` 时：

\[
w_{gan}(e)=
\begin{cases}
0, & e<50 \\
0.02\times\min\left(1,\frac{e-50+1}{50}\right), & e\ge 50
\end{cases}
\]

| Epoch | `w_gan` | 含义 |
|---|---:|---|
| 0–49 | 0 | 仅分割损失，判别器不更新 |
| 50 | 0.0004 | 开始渐进加入对抗约束 |
| 74 | 0.0100 | 达到一半基础权重 |
| 99 及以后 | 0.0200 | 使用完整 GAN 权重 |

物理一致性权重同步升温：`w_phys(e)=0.1 × w_gan(e)/0.02`。例如 epoch 50 时为 `0.002`，epoch 74 时为 `0.05`，epoch 99 后为 `0.1`。

## 5. 判别器损失：LSGAN + 物理一致性

真实标签采用单侧平滑 `0.9`，生成标签为 `0`。令 `D_r`、`D_f` 为判别器最终 Patch 评分，`P_r`、`P_f` 为物理不一致图：

\[
\mathcal{L}_{D}^{GAN} = \operatorname{MSE}(D_r, 0.9) + \operatorname{MSE}(D_f, 0)
\]

\[
\mathcal{L}_{D}^{phys} = \operatorname{MSE}(P_r, 0) + \operatorname{MSE}(P_f, 0.5)
\]

\[
\mathcal{L}_{D} = 1.0\mathcal{L}_{D}^{GAN} + 0.5\mathcal{L}_{D}^{phys}
\]

真实分割为标注转化得到的 5 通道 one-hot；生成分割使用 straight-through hard one-hot，使判别器前向看到同类数据表示，同时仍向生成器反传梯度。

在 D 更新中，真实和生成样本的分割、Depth、IR 均加入标准差 `0.05` 的 instance noise；Depth/IR 随后裁剪至 `[0,1]`。这使 D 不会依赖过于精确的输入数值差异作为真假捷径。

## 6. 生成器总损失

生成器更新时判别器参数被冻结。对抗目标将生成样本判为真（目标为 1），物理目标将生成样本的物理不一致压向 0：

\[
\mathcal{L}_{G}^{GAN}=\operatorname{MSE}(D_f,1),\qquad
\mathcal{L}_{G}^{phys}=\operatorname{MSE}(P_f,0)
\]

\[
\mathcal{L}_{G}=\mathcal{L}_{seg}+w_{gan}(e)\mathcal{L}_{G}^{GAN}+w_{phys}(e)\mathcal{L}_{G}^{phys}
\]

其中物理一致性权重与 GAN 同步升温：

\[
w_{phys}(e)=0.1\times\frac{w_{gan}(e)}{0.02}
\]

因此 epoch 0–49 时仅有 `L_seg`；epoch 99 后的完整权重为 `0.02`（GAN）和 `0.1`（物理一致性）。

G 更新时，真实物理表征在无梯度模式下由真实 one-hot 分割获得；生成分割经 straight-through one-hot 和 instance noise 进入已冻结 D。D 的参数不更新，但其对生成分割的梯度会传回 G。

## 7. 优化与学习率设置

| 项目 | 生成器 | 判别器 |
|---|---:|---:|
| 优化器 | AdamW | AdamW |
| 学习率 | `3e-4` | `1e-4` |
| betas | `(0.5, 0.999)` | `(0.5, 0.999)` |
| weight decay | `0.005` | `0.005` |
| 更新比例 | 1 step | 1 step |
| 梯度裁剪 | L2 norm ≤ `2.0` | L2 norm ≤ `2.0` |
| AMP | 启用 | 启用 |

学习率策略为 5 个 epoch 的线性 warmup，之后执行余弦退火：`T_max=400`，最小学习率 `5e-7`。最大训练 epoch 为 400；验证每 epoch 一次，验证 mIoU 连续 100 次未提升时早停。

推理阶段只执行 `Depth + IR → Generator → argmax`；不计算任何 GAN、物理一致性或判别器分支。
