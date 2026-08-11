# 实验 7 生成器：DeepLabV3+

本文仅记录实验 7 `DeepLabV3++GAN` 的生成器 `DeepLabV3Plus`，用于后续绘制网络图。实验 7 不包含 TOFAA、RGST、CBPR、边界分支、DropBlock 或 early/mid/late 物理融合。

## 1. 模型规模与输入输出

| 项目 | 实验 7 设置 |
|---|---|
| 输入 | `B × 2 × 576 × 640`，通道顺序 `[Depth, IR]` |
| 输出 | `B × 5 × 576 × 640`，未归一化 logits |
| 类别 | Background、VMPET、PI、SCTP、BCC |
| Backbone | ResNet-50 风格，block 数 `3 / 4 / 6 / 3` |
| ASPP | 256 通道；空洞率 `6 / 12 / 18` |
| Decoder | 低层投影 48 通道、融合 256 通道 |
| 可训练参数 | **40,344,933** |
| FP32 参数存储 | **153.904 MiB** |

| 子模块 | 参数量 |
|---|---:|
| ResNet Backbone | 23,504,896 |
| ASPP | 15,535,104 |
| Decoder + 5 类分类头 | 1,304,933 |
| 合计 | **40,344,933** |

## 2. 绘图主数据流

```text
Depth + IR
B×2×576×640
       │
       ▼
ResNet-50-style Backbone
 ├─ low-level F_low:  B×256×144×160 ──────────────────────┐
 └─ high-level F_high: B×2048×18×20                       │
                    │                                      │
                    ▼                                      │
             ASPP: B×256×18×20                             │
                    │                                      │
                    └─ bilinear upsample → B×256×144×160  │
                                                           ▼
                              Decoder fusion + classifier
                                                           │
                                                           ▼
                              5-class logits: B×5×576×640
```

输出保持 logits；训练时直接输入分割损失，推理时以 `argmax(dim=1)` 得到类别图。

## 3. Backbone

| 阶段 | 结构 | 输出尺寸 |
|---|---|---|
| Stem | `7×7 Conv(2→64, s=2) + BN + ReLU` | `B×64×288×320` |
| Pool | `3×3 MaxPool(s=2)` | `B×64×144×160` |
| layer1 | 3 × Bottleneck，内部 64、输出 256 | `B×256×144×160` |
| layer2 | 4 × Bottleneck，首块 `s=2`，输出 512 | `B×512×72×80` |
| layer3 | 6 × Bottleneck，首块 `s=2`，输出 1024 | `B×1024×36×40` |
| layer4 | 3 × Bottleneck，首块 `s=2`，输出 2048 | `B×2048×18×20` |

每个 Bottleneck 为：

```text
1×1 Conv → BN → ReLU → 3×3 Conv → BN → ReLU → 1×1 Conv → BN
                                      │
                           identity / 1×1 projection shortcut
                                      │
                                      └── Add → ReLU
```

`layer1` 的 `B×256×144×160` 特征直接送入 Decoder，是实验 7 的低层细节路径。

## 4. ASPP

输入 `F_high: B×2048×18×20`。五条并行分支都输出 256 通道：

| 分支 | 操作 |
|---|---|
| 1 | `1×1 Conv(2048→256) + BN + ReLU` |
| 2 | `3×3 Atrous Conv(d=6, 2048→256) + BN + ReLU` |
| 3 | `3×3 Atrous Conv(d=12, 2048→256) + BN + ReLU` |
| 4 | `3×3 Atrous Conv(d=18, 2048→256) + BN + ReLU` |
| 5 | GlobalAvgPool → `1×1 Conv(2048→256) + BN + ReLU` → resize |

五路拼接为 `B×1280×18×20`，经：

```text
1×1 Conv(1280→256) + BN + ReLU + Dropout2d(0.1)
```

得到 `F_aspp: B×256×18×20`。

## 5. Decoder 与分类头

| 路径 | 操作 | 输出尺寸 |
|---|---|---|
| 低层投影 | `1×1 Conv(256→48) + BN + ReLU` | `B×48×144×160` |
| 高层对齐 | `F_aspp` 双线性上采样至 `144×160` | `B×256×144×160` |
| 特征融合 | Concat → `3×3 Conv(304→256) + BN + ReLU + Dropout2d(0.1)` | `B×256×144×160` |
| 分类头 | `3×3 Conv(256→256) + BN + ReLU → 1×1 Conv(256→5)` | `B×5×144×160` |
| 输出恢复 | 双线性上采样 ×4 | `B×5×576×640` |

绘图时应突出两条 Decoder 输入：来自 `layer1` 的低层细节与来自 ASPP 的高层语义；实验 7 没有任何附加注意力模块插在两者之间。
