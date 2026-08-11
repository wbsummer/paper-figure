# 实验 7：DeepLabV3++GAN

本文仅记录实验 7 的网络、模型规模和训练损失，供后续绘制模型框架图。实验 7 是无注意力的 DeepLabV3+ 条件 GAN 基线；不包含 TOFAA、RGST、CBPR 或边界加权 CE。

## 1. 网络组成与模型规模

| 网络 | 作用 | 参数量 | FP32 参数存储 | 推理期 |
|---|---|---:|---:|---|
| Generator：DeepLabV3+ | `[Depth, IR] → 5 类 logits` | **40,344,933** | **153.904 MiB** | 保留 |
| Discriminator：TOFPhysicsDiscriminator | 判断分割与当前 Depth/IR 是否匹配 | **803,459** | **3.065 MiB** | 不使用 |
| 训练期合计 | G + D | **41,148,392** | **156.969 MiB** | — |

判别器总参数包含一个未进入 `forward()` 的 `material_boundary` 模块（20,160 参数）；实际训练前向使用 783,299 个判别器参数。参数存储不包括梯度、AdamW 状态和激活。

## 2. 整体训练数据流

```text
Depth + IR: B×2×576×640
          │
          ▼
Generator: DeepLabV3+
          │
          ├─ logits z: B×5×576×640 ─► segmentation losses
          │
          └─ straight-through hard one-hot: B×5×576×640 ─┐
Depth + IR ────────────────────────────────────────────────┼─► Conditional Discriminator
GT mask ─► one-hot: B×5×576×640 ───────────────────────────┘
```

生成器为 ResNet-50 风格 Backbone、ASPP 与低层跳连 Decoder：

```text
B×2×576×640
 → Backbone: layer1 B×256×144×160, layer4 B×2048×18×20
 → ASPP: B×256×18×20
 → Decoder + layer1 skip
 → B×5×576×640 logits
```

判别器主路径输入为 `[one-hot segmentation(5), Depth(1), IR(1)]`，即 `B×7×576×640`；输出 Patch 评分图 `B×1×71×79`，并叠加同尺度的物理不一致图。

## 3. 训练设置

| 项目 | 实验 7 设置 |
|---|---:|
| 批大小 | 8 |
| 最大 epoch | 400 |
| 优化器 | AdamW，weight decay `0.005` |
| G / D 学习率 | `3e-4` / `1e-4` |
| betas | `(0.5, 0.999)` |
| 调度 | 5 epoch 线性 warmup + cosine，最小学习率 `5e-7` |
| AMP | 启用 |
| 梯度裁剪 | L2 norm ≤ `2.0` |
| D steps per G | 1 |
| 验证频率 | 每 epoch 1 次 |
| 早停耐心值 | 50 次验证未提升 |

训练集使用同步水平翻转（0.5）、同步旋转（0.3，±10°）和传感器观测驱动噪声；验证与推理不使用数据增强或传感器噪声。

## 4. 生成器损失

令生成器 logits 为 `z`，标签为 `y`。实验 7 的分割项为：

\[
\mathcal{L}_{seg}=1.0\mathcal{L}_{Dice}+0.5\mathcal{L}_{Focal}+1.0\mathcal{L}_{CE}^{LS}.
\]

| 分割项 | 设置 |
|---|---|
| Dice | 权重 `1.0`，5 类 softmax Dice |
| Focal | 权重 `0.5`，`gamma=2.0` |
| CE | 权重 `1.0`，标签平滑 `0.05` |
| 类别权重 | 不启用（`null`） |
| L1 / Boundary / Lovász | 均关闭，权重 `0` |
| CBPR / 边界加权 CE | 均关闭 |
| ignore index | `255` |

GAN 在前 50 个 epoch 不参与生成器优化，之后 50 个 epoch 线性升温：

\[
w_{gan}(e)=
\begin{cases}
0,&e<50\\
0.05\min\left(1,\frac{e-50+1}{50}\right),&e\ge50.
\end{cases}
\]

物理一致性项与 GAN 同步升温：

\[
w_{phys}(e)=0.1\frac{w_{gan}(e)}{0.05}.
\]

生成器总损失：

\[
\mathcal{L}_G=\mathcal{L}_{seg}+w_{gan}\mathcal{L}_{G}^{GAN}+w_{phys}\mathcal{L}_{G}^{phys}.
\]

## 5. 判别器损失与目标

实验使用 LSGAN：真实标签采用单侧平滑 `0.9`，生成标签为 `0`；生成器希望生成结果被判为真实，即目标为 `1`。

\[
\mathcal{L}_{D}^{GAN}=\operatorname{MSE}(D_r,0.9)+\operatorname{MSE}(D_f,0),
\]

\[
\mathcal{L}_{D}^{phys}=\operatorname{MSE}(P_r,0)+\operatorname{MSE}(P_f,0.5),
\]

\[
\mathcal{L}_{D}=1.0\mathcal{L}_{D}^{GAN}+0.5\mathcal{L}_{D}^{phys}.
\]

\[
\mathcal{L}_{G}^{GAN}=\operatorname{MSE}(D_f,1),
\qquad
\mathcal{L}_{G}^{phys}=\operatorname{MSE}(P_f,0).
\]

训练时 D 的输入加入 `std=0.05` instance noise；G 更新时冻结 D 的参数和 BatchNorm/GroupNorm 状态，但保留从 D 输入回到生成 logits 的梯度。推理仅执行：

```text
Depth + IR → Generator logits → argmax → material mask
```
