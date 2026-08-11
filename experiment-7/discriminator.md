# 实验 7 判别器：TOF 条件物理一致性 PatchGAN

本文仅记录实验 7 `DeepLabV3++GAN` 的训练期判别器 `TOFPhysicsDiscriminator`。推理阶段只使用生成器，判别器不参与推理。

## 1. 模型规模与输入输出

| 项目 | 实验 7 设置 |
|---|---|
| 分割输入 | `B×5×576×640`，真实/生成的 5 类 one-hot 分割图 |
| ToF 条件 | Depth、IR，各为 `B×1×576×640` |
| 主路径输入 | `[Segmentation, Depth, IR] = B×7×576×640` |
| `base_channels / n_layers` | `64 / 3` |
| 归一化 | GroupNorm，`num_groups=16`；首层无归一化 |
| 物理分支尺度 | `physics_downsample=4`，即 `144×160` |
| 输出 | Patch logits 与物理不一致图，均为 `B×1×71×79` |
| 实例化参数 | **803,459** |
| FP32 参数存储 | **3.065 MiB** |

| 子系统 | 参数量 | 实际前向调用 |
|---|---:|---|
| 条件 PatchGAN 主路径 | 667,841 | 是 |
| Depth / IR 物理编码器 | 38,016 | 是 |
| 分割到物理表征预测器 | 77,442 | 是 |
| `material_boundary` 特征提取器 | 20,160 | 否 |
| 合计 | **803,459** | — |

`material_boundary` 会被类实例化但当前 `forward()` 不调用；绘制实际计算图时应省略它。实际执行的参数量为 **783,299**。

## 2. 绘图主数据流

```text
Real/Fake segmentation one-hot: B×5×576×640 ─┐
Depth: B×1×576×640 ───────────────────────────┼─► conditional PatchGAN ─► D_patch
IR:    B×1×576×640 ───────────────────────────┘                          B×1×71×79

Segmentation ─► segmentation-to-physics predictor ─┐
Depth ────────► depth encoder ──────────────────────┼─► |difference| ─► P
IR ───────────► IR encoder ─────────────────────────┘                  B×1×71×79

D_final = D_patch + 0.3 × P
```

`D_final` 未经过 sigmoid。实验 7 使用 LSGAN，故输出是回归评分图而非概率图。

## 3. 条件 PatchGAN 主路径

| 层 | 结构 | 输出尺寸 |
|---|---|---|
| 输入 | 拼接 segmentation、Depth、IR | `B×7×576×640` |
| Conv 1 | `4×4 Conv(7→64, s=2, p=1) + LeakyReLU(0.2)` | `B×64×288×320` |
| Conv 2 | `4×4 Conv(64→128, s=2, p=1) + GN(16) + LeakyReLU(0.2)` | `B×128×144×160` |
| Conv 3 | `4×4 Conv(128→256, s=2, p=1) + GN(16) + LeakyReLU(0.2)` | `B×256×72×80` |
| 输出层 | `4×4 Conv(256→1, s=1, p=1)` | `B×1×71×79` |

相邻 Patch 评分中心相隔 8 个输入像素，单个评分的理论感受野约为 `46×46` 像素。

## 4. 物理一致性支路

Depth、IR 与分割图先缩放至 `144×160`。

| 子路径 | 结构 | 输出尺寸 |
|---|---|---|
| Depth encoder | `3×3 Conv(1→32) + GN + LeakyReLU → 3×3 Conv(32→64) + GN + LeakyReLU` | `B×64×144×160` |
| IR encoder | 与 Depth encoder 相同 | `B×64×144×160` |
| 真实 ToF 表征 | 两支分别作通道均值，再拼接 | `B×2×144×160` |
| 分割物理预测器 | `3×3 Conv(5→64) + GN + LeakyReLU → 3×3 Conv(64→128) + GN + LeakyReLU → 1×1 Conv(128→2)` | `B×2×144×160` |
| 不一致图 | 两个表征 resize 到 `71×79` 后取绝对差并通道均值 | `P: B×1×71×79` |

\[
P=\operatorname{mean}_{channel}|q_{seg}-q_{Depth,IR}|,
\qquad D_{final}=D_{patch}+0.3P.
\]

## 5. 训练时的输入表示

| 样本 | 分割表示 | 条件输入 | 判别目标 |
|---|---|---|---:|
| Real | 标签转 5 通道 one-hot；忽略像素置零 | 同批 Depth、IR | `0.9` |
| Fake | 生成 logits 转 straight-through hard one-hot | 同批 Depth、IR | `0.0` |

Fake 在前向是 hard one-hot，使 D 不会仅凭“真实 one-hot、生成 softmax”区分真假；其反向梯度仍沿生成器 softmax 概率传回。D 更新时，真实与生成分割、Depth、IR 均加入标准差 `0.05` 的 instance noise，Depth/IR 随后裁剪到 `[0,1]`。
