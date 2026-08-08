# 实验 12 判别器：TOF 条件物理一致性 PatchGAN

本文仅描述实验 12 训练阶段的判别器 `TOFPhysicsDiscriminator`。它同时看到分割图、Depth 与 IR，判断“该分割是否与当前 ToF 观测匹配”。推理阶段不创建、不调用该网络。实验 12 的版本 7 更新只修复生成器 RGST 限幅的零残差反向传播；判别器结构、参数量、输入和损失均未改动。

## 1. 生效配置与模型规模

| 项目 | 实验 12 实际值 |
|---|---|
| 分割输入 | `B×5×576×640`，真实或生成的 one-hot 分割图 |
| 条件输入 | Depth、IR，各为 `B×1×576×640` |
| 主路径输入 | 拼接为 `B×7×576×640` |
| `base_channels / n_layers` | `64 / 3` |
| 归一化 | GroupNorm，`num_groups=16`；首个卷积层无归一化 |
| 物理分支下采样 | `physics_downsample=4`，即 `144×160` |
| 输出 | 最终 Patch 评分图和物理不一致图，均为 `B×1×71×79` |
| 可训练参数量 | **803,459** |
| FP32 参数存储 | **3.065 MiB** |

| 参数子系统 | 参数量 | 训练前向是否调用 |
|---|---:|---|
| TOF 条件 PatchGAN 主路径 | 667,841 | 是 |
| Depth + IR 真实物理编码器 | 38,016 | 是 |
| 分割→物理表征预测器 | 77,442 | 是 |
| `material_boundary` 特征提取器 | 20,160 | 否 |
| 实例化总计 | **803,459** | — |

`material_boundary` 在当前类中会被实例化并计入总参数，但 `forward()` 不调用它。后续绘图应省略该支路，并明确主计算图实际使用 783,299 个参数。

## 2. 总体数据流

```text
Real/Fake segmentation one-hot: B×5×576×640 ─┐
Depth: B×1×576×640 ───────────────────────────┼─► conditional PatchGAN ─► patch logits
IR:    B×1×576×640 ───────────────────────────┘                         B×1×71×79

Segmentation ─► segmentation-to-physics predictor ─┐
Depth ────────► depth encoder ──────────────────────┼─► absolute difference ─► P
IR ───────────► IR encoder ─────────────────────────┘                       B×1×71×79

Final score = patch logits + 0.3 × P
```

最终评分没有 sigmoid。实验使用 LSGAN 的均方误差，因此它是回归评分图，不应当解释为概率。

## 3. TOF 条件 PatchGAN 主路径

| 层 | 结构 | 输出张量 |
|---|---|---|
| 输入 | `[segmentation, Depth, IR]` 拼接 | `B×7×576×640` |
| Conv 1 | `4×4 Conv(7→64, s=2, p=1) + LeakyReLU(0.2)` | `B×64×288×320` |
| Conv 2 | `4×4 Conv(64→128, s=2, p=1) + GN(16) + LeakyReLU(0.2)` | `B×128×144×160` |
| Conv 3 | `4×4 Conv(128→256, s=2, p=1) + GN(16) + LeakyReLU(0.2)` | `B×256×72×80` |
| 输出层 | `4×4 Conv(256→1, s=1, p=1)` | `B×1×71×79` |

三个 stride-2 层后，Patch 评分的相邻中心相隔 8 个输入像素；单个评分的理论感受野约为 `46×46`。主路径输出记为 `D_patch`。

## 4. 物理一致性支路

Depth、IR 与分割图均被缩放至 `144×160` 后进入物理支路。

| 子路径 | 结构 | 输出张量 |
|---|---|---|
| Depth encoder | `3×3 Conv(1→32) + GN + LeakyReLU → 3×3 Conv(32→64) + GN + LeakyReLU` | `B×64×144×160` |
| IR encoder | 与 Depth encoder 相同 | `B×64×144×160` |
| 真实物理表征 | 两路特征分别沿通道均值化后拼接 | `B×2×144×160` |
| 分割物理预测器 | `3×3 Conv(5→64) + GN + LeakyReLU → 3×3 Conv(64→128) + GN + LeakyReLU → 1×1 Conv(128→2)` | `B×2×144×160` |
| 物理不一致图 | 两个 2 通道表征上采样至 `71×79` 后取绝对差并按通道均值 | `P: B×1×71×79` |

\[
P=\operatorname{mean}_{channel}\left|\operatorname{resize}(q_{seg})-\operatorname{resize}(q_{Depth,IR})\right|,
\]

\[
D_{final}=D_{patch}+0.3P.
\]

这使主判别器评分同时考虑局部分割纹理/边界模式和分割结果与当前 Depth/IR 的物理一致性。

## 5. 训练输入表示

| 样本 | 分割图前向表示 | ToF 条件 | 判别器目标 |
|---|---|---|---:|
| Real | 标注 mask 转 5 通道 one-hot；忽略像素置零 | 同一批配对 Depth、IR | `0.9` |
| Fake | 生成 logits 经 straight-through argmax one-hot | 同一批配对 Depth、IR | `0.0` |

Fake 的前向值是硬 one-hot，与 Real 的数值形式一致，避免 D 仅凭“真实是 one-hot、生成是 softmax 概率”取巧；反向梯度仍沿 softmax 概率传回 G。D 更新时，Real 与 Fake 的分割图、Depth、IR 都加入标准差 `0.05` 的 instance noise；Depth/IR 再裁剪回 `[0,1]`。
