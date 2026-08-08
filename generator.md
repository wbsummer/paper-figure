# 实验 11 生成器：DeepLabV3++GAN + RGST-TOFAA

本文仅描述实验 11 的实际生成器：`DeepLabV3Plus` + `ReliabilityGatedSemanticTOFAttention`（RGST-TOFAA）。训练时由 `ablation_study.py` 将生成器设置为 `deeplabv3plus`、`fusion_type=none`、`attention_type=reliability_gated_semantic_tofaa`。

## 模型规模

| 项目 | 数值 |
|---|---:|
| 输入 | `B x 2 x 576 x 640`，通道为 `[Depth, IR]` |
| 输出 | `B x 5 x 576 x 640`，未归一化类别 logits |
| 类别 | Background、VMPET、PI、SCTP、BCC |
| 总参数量 | **41,094,343** |
| FP32 参数存储 | **156.762 MiB** |
| Backbone | 23,504,896 参数，89.664 MiB |
| ASPP | 15,535,104 参数，59.262 MiB |
| RGST-TOFAA | 749,410 参数，2.859 MiB |
| Decoder | 1,304,933 参数，4.978 MiB |

> 模型大小仅指可训练参数的 FP32 存储，不包含梯度、优化器状态和中间激活。

## 实验 11 生效配置

| 配置项 | 生效值 | 作用 |
|---|---|---|
| `generator_type` | `deeplabv3plus` | 选择本文件描述的生成器 |
| `in_channels / out_channels` | `2 / 5` | 输入 Depth、IR；输出背景加 4 种材料 |
| `fusion_type` | `none` | 不实例化 early / mid / late 物理融合支路 |
| `use_attention` | `true` | 启用 RGST-TOFAA |
| `attention_type` | `reliability_gated_semantic_tofaa` | 仅在 ASPP 后做高层语义校正 |
| `tofaa_implementation_version` | `5` | 与 V2/V3/V4 checkpoint 隔离 |
| `use_dropblock` | `false` | 不使用可选 DropBlock；ASPP/Decoder 自带的 `Dropout2d(0.1)` 仍保留 |

输入预处理后的 Depth 与 IR 均在 `[0,1]`：Depth 先乘 `0.02` m、裁剪到 `5.1` m 再归一化；IR 除以 `255` 后裁剪。生成器没有 sigmoid 或 softmax 输出层，训练损失直接接收 logits。

## 整体结构与张量尺度

```text
Depth + IR (B x 2 x 576 x 640)
        │
        ▼
ResNet-50-style Backbone
        ├──────────────────────────────► layer1: B x 256 x 144 x 160 ──────┐
        ▼                                                                  │
layer4: B x 2048 x 18 x 20                                                 │
        │                                                                  │
        ▼                                                                  │
ASPP: B x 256 x 18 x 20 → RGST-TOFAA: B x 256 x 18 x 20 ──────────────────┤
                                                                           ▼
                                                        DeepLabV3+ Decoder
                                                                           │
                                                                           ▼
                                                   5-class logits: B x 5 x 576 x 640
```

## 1. ResNet 风格 Backbone

| 阶段 | 结构 | 输出尺寸 |
|---|---|---|
| Stem | `7x7 Conv(2→64, s=2) + BN + ReLU` | `B x 64 x 288 x 320` |
| Pool | `3x3 MaxPool(s=2)` | `B x 64 x 144 x 160` |
| layer1 | 3 个 bottleneck | `B x 256 x 144 x 160` |
| layer2 | 4 个 bottleneck，首块下采样 | `B x 512 x 72 x 80` |
| layer3 | 6 个 bottleneck，首块下采样 | `B x 1024 x 36 x 40` |
| layer4 | 3 个 bottleneck，首块下采样 | `B x 2048 x 18 x 20` |

每个 bottleneck 的主分支为 `1x1 Conv → BN → ReLU → 3x3 Conv → BN → ReLU → 1x1 Conv → BN`，再与恒等/`1x1 Conv + BN` 残差支路相加后 ReLU。四个 stage 的 bottleneck 中间通道分别为 `64 / 128 / 256 / 512`，输出通道按 expansion=4 变为 `256 / 512 / 1024 / 2048`。

实验 11 的 RGST-TOFAA **不修改** `layer1` 特征；该原始低层路径直接送入 Decoder，保留空间边界信息。

## 2. ASPP 高层语义提取

输入为 `B x 2048 x 18 x 20`。五个并行分支均输出 256 通道：

- `1x1 Conv`；
- `3x3 Atrous Conv`，dilation = 6；
- `3x3 Atrous Conv`，dilation = 12；
- `3x3 Atrous Conv`，dilation = 18；
- 全局平均池化 → `1x1 Conv` → 上采样。

五路特征拼接为 1280 通道，经 `1x1 Conv(1280→256) + BN + ReLU + Dropout2d(0.1)`，得到 ASPP 特征 `F`：`B x 256 x 18 x 20`。

| ASPP 子模块 | 参数量 |
|---|---:|
| `1x1` 分支 | 524,800 |
| 3 个空洞卷积分支 | 14,157,312 |
| 全局池化分支 | 524,800 |
| 1280→256 融合层 | 328,192 |
| 合计 | **15,535,104** |

## 3. RGST-TOFAA：可靠性门控语义 TOF 聚合

RGST-TOFAA 位于 ASPP 之后，仅校正高层语义；Depth 与 IR 被双线性下采样到 `18 x 20`。

```text
Depth, IR, |∇Depth|, |∇IR|  (B x 4 x 18 x 20)
            │
            ▼
TOF encoder: 3x3 Conv(4→32) → BN → ReLU → 3x3 Conv(32→32) → BN → ReLU
            │  T: B x 32 x 18 x 20
ASPP F ─────┴── concat ──► [F, T]: B x 288 x 18 x 20
                              ├─ Reliability gate: 1x1 Conv(288→32→1) + Sigmoid → g
                              └─ Semantic delta: 3x3 Conv(288→256) + BN + ReLU → 1x1 Conv(256→256) → ΔF

F_out = F + tanh(α) × g × ΔF
```

其中 `g` 为逐像素可靠性门控图，尺寸 `B x 1 x 18 x 20`；`α` 是可学习标量，初始化为 0。因此训练开始时 `F_out = F`，实验 11 的原 Backbone、ASPP 和 Decoder 初始输出与实验 7 完全一致；仅当训练证明 TOF 校正有效时才逐渐引入 `ΔF`。

有限差分梯度在水平方向和垂直方向分别计算相邻像素差的绝对值，再相加；右边界和下边界补零。因此 RGST 的 TOF 观测完全由当前 Depth/IR 输入导出，不使用材质标签或预测标签。

| RGST 子模块 | 参数量 | 输出 |
|---|---:|---|
| TOF encoder | 10,496 | `T: B x 32 x 18 x 20` |
| reliability gate | 9,313 | `g: B x 1 x 18 x 20` |
| semantic delta | 729,600 | `ΔF: B x 256 x 18 x 20` |
| 标量 `α` | 1 | 单个可学习尺度 |
| 合计 | **749,410** | `F_out: B x 256 x 18 x 20` |

`tanh(α)` 将残差全局尺度限制在 `(-1,1)`。由于 `α=0`，第一步仅 `α` 能通过非零的 `g×ΔF` 获得梯度；当其离开零后，门控和校正卷积分支才同时收到有效梯度。这一设计确保新模块不会在初始化时扰动实验 7 的分割路径。

## 4. Decoder 与分类头

| 路径 | 操作 | 输出尺寸 |
|---|---|---|
| 低层路径 | `1x1 Conv(256→48) + BN + ReLU` | `B x 48 x 144 x 160` |
| 高层路径 | `F_out` 上采样到 `144 x 160` | `B x 256 x 144 x 160` |
| 融合 | 拼接 → `3x3 Conv(304→256) + BN + ReLU + Dropout2d(0.1)` | `B x 256 x 144 x 160` |
| 分类头 | `3x3 Conv(256→256) + BN + ReLU → 1x1 Conv(256→5)` | `B x 5 x 144 x 160` |
| 输出 | 双线性上采样 4 倍 | `B x 5 x 576 x 640` |

推理时对 logits 使用 `argmax(dim=1)` 得到最终材质标签图。

| Decoder 子模块 | 参数量 |
|---|---:|
| 低层 `256→48` 投影 | 12,384 |
| 低/高层融合卷积 | 700,928 |
| `256→256→5` 分类头 | 591,621 |
| 合计 | **1,304,933** |

## 5. 实际前向顺序

1. 输入 `x=[Depth,IR]` 经 Backbone 得到原始低层 `f_low` 与高层 `f_high`。
2. `f_high` 经 ASPP 得到 `F`；原始 Depth、IR 下采样至 `18 x 20` 后与 `F` 一起进入 RGST-TOFAA，输出 `F_out`。
3. **未经任何注意力改写的** `f_low` 经 `256→48` 投影；`F_out` 上采样至同一 `144 x 160` 尺度。
4. 两路拼接、卷积融合、分类，再上采样 4 倍得到 5 类 logits。

模型构造时 RGST-TOFAA 在 Backbone、ASPP 和 Decoder 后才实例化；这避免其参数初始化消耗随机数后改变实验 7 主干的同种子初始化。RGST-TOFAA 是生成器的一部分，训练和推理均会执行；判别器只在训练期使用。
