# 实验 11 判别器：TOF 条件物理一致性 PatchGAN

本文仅描述实验 11 的实际判别器 `TOFPhysicsDiscriminator`。判别器仅在训练阶段参与对抗学习和物理一致性约束；推理阶段不创建、不调用判别器。

## 模型规模

| 项目 | 数值 |
|---|---:|
| 分割输入 | `B x 5 x 576 x 640`，真实或生成的 one-hot 分割图 |
| 条件输入 | Depth、IR，各为 `B x 1 x 576 x 640` |
| 主路径输入 | 拼接后 `B x 7 x 576 x 640` |
| Patch 层数 | 3 个 stride-2 卷积层 + 1 个输出卷积层 |
| 归一化 | GroupNorm，最多 16 组；首层无归一化 |
| 物理分支尺度 | 输入下采样 4 倍，即 `144 x 160` |
| 输出 | Patch 评分图、物理不一致图，均为 `B x 1 x 71 x 79` |
| 总参数量 | **803,459** |
| FP32 参数存储 | **3.065 MiB** |

> 模型大小仅指当前实例化判别器的 FP32 参数，不包含梯度、优化器状态和中间激活。
> 当前源码还会实例化一个未被 `forward()` 调用的 `material_boundary` 提取器（20,160 参数）；为保证框图反映实际计算路径，本文不绘制该支路。实际前向路径使用 783,299 参数。

| 参数子系统 | 参数量 | 是否进入实验 11 `forward()` |
|---|---:|---|
| 条件 PatchGAN 主路径 | 667,841 | 是 |
| Depth / IR 真实物理编码器 | 38,016 | 是 |
| 分割物理预测器 | 77,442 | 是 |
| `material_boundary` 提取器 | 20,160 | 否 |
| 实例化总计 | **803,459** | — |

## 整体结构

```text
Segmentation one-hot (B x 5 x 576 x 640) ─┐
Depth (B x 1 x 576 x 640) ────────────────┼─► Conditional PatchGAN ─► Patch score
IR    (B x 1 x 576 x 640) ────────────────┘

Segmentation one-hot ─► Physics predictor ─┐
Depth ─────────────────► Depth encoder ────┼─► absolute difference ─► physics score
IR ────────────────────► IR encoder ───────┘

Final score = Patch score + 0.3 × physics score
```

## 1. TOF 条件 PatchGAN 主路径

| 层 | 结构 | 输出尺寸 |
|---|---|---|
| 输入 | `[segmentation, Depth, IR]` 拼接 | `B x 7 x 576 x 640` |
| Conv 1 | `4x4 Conv(7→64, s=2, p=1) + LeakyReLU(0.2)` | `B x 64 x 288 x 320` |
| Conv 2 | `4x4 Conv(64→128, s=2, p=1) + GN(16) + LeakyReLU(0.2)` | `B x 128 x 144 x 160` |
| Conv 3 | `4x4 Conv(128→256, s=2, p=1) + GN(16) + LeakyReLU(0.2)` | `B x 256 x 72 x 80` |
| 输出层 | `4x4 Conv(256→1, s=1, p=1)` | `B x 1 x 71 x 79` |

每个评分单元对应一个重叠局部区域，因此输出是 PatchGAN 评分图，而不是单个全图真假标量。

三个 stride-2 卷积后，输出层每个 Patch 评分对应原输入约 `46 x 46` 的感受野，Patch 间步长为 8 像素。主路径各卷积参数量依次为 `7,232 / 131,456 / 525,056 / 4,097`，合计 667,841。

## 2. 物理一致性支路

Depth、IR 与分割图先缩放至 `144 x 160`。

| 分支 | 结构 | 输出尺寸 |
|---|---|---|
| Depth encoder | `3x3 Conv(1→32) + GN + LeakyReLU → 3x3 Conv(32→64) + GN + LeakyReLU` | `B x 64 x 144 x 160` |
| IR encoder | 与 Depth encoder 相同 | `B x 64 x 144 x 160` |
| 真实物理表征 | 两路特征分别沿通道求均值后拼接 | `B x 2 x 144 x 160` |
| 分割物理预测器 | `3x3 Conv(5→64) + GN + LeakyReLU → 3x3 Conv(64→128) + GN + LeakyReLU → 1x1 Conv(128→2)` | `B x 2 x 144 x 160` |
| 物理评分 | 两个 2 通道表征上采样至 `71 x 79` 后取逐元素绝对差并沿通道平均 | `B x 1 x 71 x 79` |

最终判别器输出为：

```text
D(segmentation, Depth, IR) = patch_score + 0.3 × physics_score
```

该输出用于 LSGAN 的均方误差损失，不经过 sigmoid，不能解释为概率。

实际的物理一致性图为：

\[
P=\operatorname{mean}_{channel}\left|\operatorname{resize}(\hat{q}_{seg})-\operatorname{resize}(q_{Depth,IR})\right|
\]

其中 `q_Depth,IR` 来自真实 Depth/IR 编码器，`q_hat_seg` 由分割图预测。二者均被插值到 Patch 评分的 `71 x 79` 尺度后比较。`P` 同时作为独立物理损失的输入，并以系数 `0.3` 加到主 Patch 评分上。

## 3. 训练时输入表示

| 样本 | 分割输入 | Depth / IR 条件 | 用途 |
|---|---|---|---|
| Real | 标注 mask 转 5 通道 one-hot | 当前配对 Depth、IR | D 学习真实目标 `0.9` |
| Fake | 生成器 logits 转 straight-through hard one-hot | 同一对 Depth、IR | D 学习生成目标 `0`；G 仍可接收梯度 |

straight-through one-hot 的前向值是 argmax one-hot，而反向值沿 softmax 概率传播。因此 D 不会仅靠“真实是 one-hot、生成是连续概率”这种数值捷径区分样本。
