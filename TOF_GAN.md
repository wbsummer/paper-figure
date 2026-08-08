# 实验 12：DeepLabV3++GAN + RGST-TOFAA-RC

本文记录实验 12 的实际网络、参数规模、训练数据流与损失搭建，供后续绘制模型框架图和撰写方法章节使用。实验 12 在实验 7 的强分割基线之上，加入位于 ASPP 后的 RGST-TOFAA-RC；其核心约束是最终语义残差的 RMS 不超过原 ASPP 特征 RMS 的 10%。

## 1. 网络组成与规模

| 网络 | 功能 | 参数量 | FP32 参数存储 | 推理期 |
|---|---|---:|---:|---|
| Generator：DeepLabV3+ + RGST-TOFAA-RC | Depth/IR → 5 类 logits | **41,094,343** | **156.762 MiB** | 保留 |
| Discriminator：TOFPhysicsDiscriminator | 判断分割是否匹配当前 Depth/IR | **803,459** | **3.065 MiB** | 不使用 |
| 训练期合计 | G + D | **41,897,802** | **159.827 MiB** | — |

判别器总参数包含一个未进入 `forward()` 的 20,160 参数 `material_boundary` 模块；真实前向计算路径使用 783,299 个判别器参数。模型参数存储不含梯度、AdamW 状态和中间激活。

### 实验 12 版本与可训练性

实验 12 当前使用 `attention_type=reliability_gated_semantic_tofaa_rc`、`rgst_max_residual_ratio=0.1` 和 TOFAA checkpoint **版本 7**。版本 7 修复了零初始化时的数值稳定性：因为 `α=0`，初始有效残差 `C=tanh(α)gΔF=0`；RMS 使用 `sqrt(mean(C²)+10^-12)`，保证第一步反向传播的所有生成器梯度均有限。旧版本 6 在 `sqrt(mean(C²))` 的零点反向会产生 NaN，AMP 会跳过整个 G 优化步，故其 checkpoint 不可恢复、不可用于推理或结果比较。

```text
Depth + IR ─► Generator ─► logits ─► straight-through one-hot ─┐
       │                                                        │
       └────────────────────────────────────────────────────────┼─► Conditional D
Depth + IR ──────────────────────────────────────────────────────┘
```

### 生成器主干

```text
B×2×576×640
  → ResNet backbone
  → layer4: B×2048×18×20
  → ASPP: B×256×18×20
  → RGST-TOFAA-RC: B×256×18×20
  → Decoder + layer1 skip: B×5×576×640 logits
```

RGST-TOFAA-RC 输入为 ASPP 特征及对齐到 `18×20` 的 Depth/IR。它生成可靠性门控 `g` 和语义差分 `ΔF`，先计算：

\[
C=\tanh(\alpha)g\Delta F.
\]

然后按样本限幅：

\[
F_{out}=F_{ASPP}+sC,\quad
s=\min\left(1,\frac{0.1\operatorname{RMS}(F_{ASPP})}{\sqrt{\operatorname{mean}(C^2)+10^{-12}}}\right).
\]

因此 `RMS(F_out − F_ASPP) ≤ 0.1 × RMS(F_ASPP)`。低层 `layer1` 特征保持原样，经 `256→48` 投影后与 `F_out` 在 Decoder 中融合。

### 判别器主干

```text
[one-hot segmentation(5), Depth(1), IR(1)] = B×7×576×640
  → Conv 7→64, stride 2
  → Conv 64→128, stride 2 + GN
  → Conv 128→256, stride 2 + GN
  → Conv 256→1, stride 1
  → Patch logits: B×1×71×79
```

并行物理支路将 Depth、IR 与 segmentation 缩放到 `144×160`，构造真实 ToF 物理表征和分割预测物理表征，得到不一致图 `P: B×1×71×79`。最终判别器输出为 `D_final=D_patch+0.3P`。

## 2. 训练批次、增强与对抗时序

每个批次为 `B=8` 个对齐的 `(Depth, IR, mask)`，分辨率为 `576×640`。训练集开启同步水平翻转（`0.5`）、同步旋转（`0.3`，`±10°`）和传感器观测驱动噪声；验证和推理不使用增强。

当前传感器噪声配置为：

| 类型 | 配置 |
|---|---|
| diffuse | `std=0.005` |
| specular | `prob=0.5`，阈值 `0.92`，深度噪声 `std=0.005` |
| subsurface | `prob=0.4`，最大 bias `0.02` |
| multipath | `prob=0.4`，`std=0.015` |

对抗训练采用先分割、后渐进对抗的时序：

| Epoch | D 更新 | G 中的 GAN/物理项 |
|---|---|---|
| 0–49 | 不更新 | 不启用；仅分割损失 |
| 50–98 | 每个 G step 前更新 D 一次 | 线性升温 |
| 99 以后 | 每个 G step 前更新 D 一次 | 完整权重 |

`d_steps_per_g=1`。G 更新时，D 切换为 eval 且全部参数 `requires_grad=False`；D 的前向梯度仍回传到生成分割图。训练启用 AMP、L2 梯度裁剪（最大范数 `2.0`）和 AdamW。

## 3. 生成器分割损失

令生成器 logits 为 `z`、标签为 `y`，分割损失为：

\[
\mathcal{L}_{seg}=1.0\mathcal{L}_{Dice}+0.5\mathcal{L}_{Focal}+1.0\mathcal{L}_{CE}^{LS}.
\]

| 项 | 实际设置 |
|---|---|
| Dice | 权重 `1.0`，5 类 softmax Dice 平均 |
| Focal | 权重 `0.5`，`gamma=2.0`；5 类任务中 `focal_alpha=0.25` 不参与二分类式 alpha 加权 |
| Label-smoothed CE | 权重 `1.0`，标签平滑 `0.05` |
| 类别权重 | 未启用（`null`） |
| L1 / Boundary / Lovász | 均关闭，权重 `0` |
| ignore index | `255` |

## 4. GAN 与物理一致性损失

基础生成器对抗权重为 `lambda_gan=0.02`，开始 epoch 为 50，升温长度为 50：

\[
w_{gan}(e)=
\begin{cases}
0,&e<50\\
0.02\min\left(1,\frac{e-50+1}{50}\right),&e\ge50.
\end{cases}
\]

生成器物理一致性权重与其同步升温：

\[
w_{phys}(e)=0.1\frac{w_{gan}(e)}{0.02}.
\]

在 LSGAN 下，D 的真实标签为 `0.9`（单侧标签平滑），Fake 标签为 `0`：

\[
\mathcal{L}_{D}^{GAN}=\operatorname{MSE}(D_r,0.9)+\operatorname{MSE}(D_f,0),
\]

\[
\mathcal{L}_{D}^{phys}=\operatorname{MSE}(P_r,0)+\operatorname{MSE}(P_f,0.5),
\]

\[
\mathcal{L}_{D}=1.0\mathcal{L}_{D}^{GAN}+0.5\mathcal{L}_{D}^{phys}.
\]

G 的目标是使生成分割被判为真实且物理不一致趋近零：

\[
\mathcal{L}_{G}^{GAN}=\operatorname{MSE}(D_f,1),\qquad
\mathcal{L}_{G}^{phys}=\operatorname{MSE}(P_f,0),
\]

\[
\mathcal{L}_{G}=\mathcal{L}_{seg}+w_{gan}(e)\mathcal{L}_{G}^{GAN}+w_{phys}(e)\mathcal{L}_{G}^{phys}.
\]

Real 分割来自标签 one-hot；Fake 分割使用 straight-through hard one-hot：D 前向看到 argmax one-hot，反向梯度沿 softmax 传回 G。D 更新时，Real/Fake 分割与对应 Depth/IR 均添加 `std=0.05` instance noise；G 更新时仅 Fake 输入进入这一加噪路径。

## 5. 优化与停止准则

| 项目 | Generator | Discriminator |
|---|---:|---:|
| 优化器 | AdamW | AdamW |
| 学习率 | `3e-4` | `1e-4` |
| betas | `(0.5, 0.999)` | `(0.5, 0.999)` |
| weight decay | `0.005` | `0.005` |
| 学习率策略 | 5 epoch 线性 warmup + cosine 至 `5e-7` | 相同 |
| 梯度裁剪 | L2 norm ≤ `2.0` | L2 norm ≤ `2.0` |

最大训练 epoch 为 400；`val_freq=1`，即每个 epoch 验证一次；当前早停为验证 mIoU 连续 100 次不提升。最终推理只使用生成器：`Depth + IR → logits → argmax`，不计算任何对抗或物理分支。
