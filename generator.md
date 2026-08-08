# 实验 12 生成器：DeepLabV3++GAN + RGST-TOFAA-RC

本文仅描述实验 12 的生成器。实际类为 `DeepLabV3Plus`，材质感知模块为 `ReliabilityGatedSemanticTOFAttention` 的相对残差限幅变体（RGST-TOFAA-RC）。它以实验 7 的 DeepLabV3+ 为主体，在 ASPP 输出之后加入 Depth/IR 条件的高层语义微调；不修改 `layer1` 的低层边界特征。

## 1. 生效配置与模型规模

| 项目 | 实验 12 实际值 |
|---|---|
| `generator_type` | `deeplabv3plus` |
| 输入 | `B × 2 × 576 × 640`，通道顺序为 `[Depth, IR]` |
| 输出 | `B × 5 × 576 × 640`，未归一化 logits |
| 类别 | Background、VMPET、PI、SCTP、BCC |
| `fusion_type` | `none`；无 early/mid/late 物理融合支路 |
| `attention_type` | `reliability_gated_semantic_tofaa_rc` |
| `rgst_max_residual_ratio` | `0.1` |
| TOFAA checkpoint 版本 | `7` |
| 可训练参数量 | **41,094,343** |
| FP32 参数存储 | **156.762 MiB** |

`rgst_max_residual_ratio=0.1` 是固定超参数，不增加可训练参数；它对最终写入 ASPP 的残差施加逐样本相对 RMS 上限。模型输出不经过 sigmoid 或 softmax；训练损失直接接收 logits，推理使用 `argmax(dim=1)`。

| 生成器子模块 | 参数量 | FP32 参数存储 |
|---|---:|---:|
| ResNet-50-style Backbone | 23,504,896 | 89.664 MiB |
| ASPP | 15,535,104 | 59.262 MiB |
| RGST-TOFAA-RC | 749,410 | 2.859 MiB |
| DeepLabV3+ Decoder + 分类头 | 1,304,933 | 4.978 MiB |
| 合计 | **41,094,343** | **156.762 MiB** |

## 2. 整体数据流与尺度

```text
Depth + IR: B×2×576×640
       │
       ▼
ResNet Backbone
 ├─ layer1 low-level: B×256×144×160 ───────────────────────────┐
 └─ layer4 high-level: B×2048×18×20                             │
                   │                                             │
                   ▼                                             │
             ASPP: B×256×18×20                                  │
                   │                                             │
        RGST-TOFAA-RC + resized Depth/IR                         │
                   │                                             │
                   ▼                                             ▼
       corrected semantic: B×256×18×20              Decoder fusion
                                                           │
                                                           ▼
                                           5-class logits: B×5×576×640
```

Depth 的预处理为原始值乘 `0.02` m、裁剪到 `5.1` m 后归一化；IR 除以 `255` 并裁剪到 `[0,1]`。RGST 使用同一批输入中的 Depth/IR，不读取标签或预测掩码。

## 3. Backbone：ResNet-50 风格编码器

| 阶段 | 结构 | 输出张量 |
|---|---|---|
| Stem | `7×7 Conv(2→64, s=2) + BN + ReLU` | `B×64×288×320` |
| Pool | `3×3 MaxPool(s=2)` | `B×64×144×160` |
| layer1 | 3 个 bottleneck，内部通道 64、输出通道 256 | `B×256×144×160` |
| layer2 | 4 个 bottleneck，首块下采样 | `B×512×72×80` |
| layer3 | 6 个 bottleneck，首块下采样 | `B×1024×36×40` |
| layer4 | 3 个 bottleneck，首块下采样 | `B×2048×18×20` |

每个 bottleneck 为 `1×1 Conv → BN → ReLU → 3×3 Conv → BN → ReLU → 1×1 Conv → BN`，然后与恒等/投影残差支路相加并 ReLU。`layer1` 特征不经过 RGST，直接作为 Decoder 的低层边界路径。

## 4. ASPP：高层多尺度语义

ASPP 接收 `B×2048×18×20`，五条分支均输出 256 通道：`1×1` 卷积、三个 `3×3` 空洞卷积（dilation `6/12/18`），以及全局平均池化分支。五路拼接为 1280 通道后，经 `1×1 Conv(1280→256) + BN + ReLU + Dropout2d(0.1)` 输出：

```text
F_ASPP: B×256×18×20
```

## 5. RGST-TOFAA-RC：可靠性门控与相对残差限幅

Depth、IR 先双线性下采样至 ASPP 尺度 `18×20`。其一阶有限差分梯度与原始观测拼接：

```text
[Depth, IR, |∇Depth|, |∇IR|]: B×4×18×20
          │
          ▼
TOF encoder: 3×3 Conv(4→32) + BN + ReLU
             3×3 Conv(32→32) + BN + ReLU
          │
          ▼
T: B×32×18×20

[F_ASPP, T]: B×288×18×20
     ├─ gate: 1×1 Conv(288→32) + BN + ReLU + 1×1 Conv(32→1) + Sigmoid
     │        g: B×1×18×20
     └─ delta: 3×3 Conv(288→256) + BN + ReLU + 1×1 Conv(256→256)
              ΔF: B×256×18×20
```

未经限幅的有效残差为：

\[
C=\tanh(\alpha)\cdot g\cdot\Delta F,
\]

其中 `α` 是一个初始化为 0 的可学习标量。因此初始时 `C=0`，实验 12 的 Backbone、ASPP 和 Decoder 初始输出与实验 7 完全一致。

实验 12 以每张样本自身的 ASPP 特征强度作为基准，在通道和空间维度上计算 RMS：

\[
r_F=\sqrt{\operatorname{mean}(F_{ASPP}^2)},\qquad
r_C=\sqrt{\operatorname{mean}(C^2)+10^{-12}}.
\]

设 `r_max = 0.1 × r_F`，最终写入 Decoder 的校正为：

\[
s=\min\left(1,\frac{r_{max}}{r_C}\right),\qquad
F_{out}=F_{ASPP}+sC.
\]

故每个样本始终满足：

\[
\operatorname{RMS}(F_{out}-F_{ASPP})\leq0.1\operatorname{RMS}(F_{ASPP}).
\]

这限制的是经过 `tanh(α)` 和可靠性门控后的**最终有效残差**，而非仅限制网络内部未调制的 `ΔF`。当残差本来较小时，`s=1`，不改变其大小；只有超过上限时才按比例缩小。

| RGST-TOFAA-RC 子模块 | 参数量 |
|---|---:|
| TOF encoder | 10,496 |
| reliability gate | 9,313 |
| semantic delta | 729,600 |
| 可学习标量 `α` | 1 |
| 合计 | **749,410** |

## 6. Decoder 与输出头

| 路径 | 操作 | 输出张量 |
|---|---|---|
| 低层路径 | `1×1 Conv(256→48) + BN + ReLU` | `B×48×144×160` |
| 高层路径 | `F_out` 双线性上采样至 `144×160` | `B×256×144×160` |
| 融合 | 拼接后 `3×3 Conv(304→256) + BN + ReLU + Dropout2d(0.1)` | `B×256×144×160` |
| 分类头 | `3×3 Conv(256→256) + BN + ReLU → 1×1 Conv(256→5)` | `B×5×144×160` |
| 输出 | 双线性上采样 4 倍 | `B×5×576×640` |

推理阶段只执行 `Depth + IR → Generator → argmax`。RGST-TOFAA-RC 是生成器的一部分，训练和推理都会执行；判别器不参与推理。
