# DeepLabV3+ Generator Figure Design

## Objective

Create a publication-ready, fully editable 2D architecture figure for the generator used in ablation experiments 6--9. The DeepLabV3+ computation path is the visual focus; TOFAA V2 and V3 appear as compact explanatory insets.

## Deliverables

- Editable vector source.
- Review PNG approximately 2000 px wide.
- English-only labels suitable for an IEEE/CVPR/ICCV manuscript.
- Wide 16:9 composition intended for a two-column page-width figure.

## Composition

The upper 70% contains a left-to-right DeepLabV3+ pipeline:

`Depth + IR -> Stem -> Layer 1 -> Layer 2 -> Layer 3 -> Layer 4 -> ASPP -> Decoder -> 5-Class Logits`

The lower 30% contains two compact insets, TOFAA V2 and TOFAA V3. Dashed callouts associate both insets with the optional low-level TOFAA position. A separate purple guidance connector associates the ASPP output with the semantic input of V3.

## Main Pipeline

- Input: `Depth + IR`, tensor `2 x 576 x 640` (batch dimension omitted for visual economy).
- Backbone container:
  - Stem: `7x7 Conv, s=2` followed by `3x3 MaxPool, s=2`.
  - Layer 1: `3 Bottlenecks`, output `256 x 144 x 160`.
  - Layer 2: `4 Bottlenecks`, output `512 x 72 x 80`.
  - Layer 3: `6 Bottlenecks`, output `1024 x 36 x 40`.
  - Layer 4: `3 Bottlenecks`, output `2048 x 18 x 20`.
- High-level path: Layer 4 enters ASPP. ASPP shows five parallel branches labelled `1x1`, `3x3 r=6`, `3x3 r=12`, `3x3 r=18`, and `Image Pooling`. Their outputs merge through `Concat + 1x1 Conv` into `256 x 18 x 20`.
- Low-level path: Layer 1 branches around Layers 2--4 through `Optional TOFAA`, then `1x1 Conv, 256->48`.
- Decoder: ASPP output is bilinearly upsampled to `144 x 160`, concatenated with the 48-channel low-level feature, processed by two `3x3 Conv` blocks and a `1x1 Classifier`, then upsampled by four.
- Output: `5-Class Logits`, tensor `5 x 576 x 640`.

## Simplified TOFAA Insets

### TOFAA V2

`Depth/IR Encoding -> Physical Gate -> Channel + Spatial Attention -> Feature Fusion -> Residual Gate`

The inset communicates that encoded depth and IR jointly modulate the Layer 1 feature and that the update is residual. Internal convolution-by-convolution detail is intentionally omitted.

### TOFAA V3

V3 repeats the V2 stages but adds `ASPP Semantic Guidance` before joint modulation. Its defining message is `Physical-Semantic Joint Modulation`.

## Visual Language

- Pure white background and flat 2D geometry.
- Backbone: muted blue; ASPP/context: violet; TOFAA/attention: coral; decoder/head: sage green; input/output: graphite/light gray.
- Dark gray 1.5--2 pt outlines, modest corner radii, and consistent filled triangular arrowheads.
- Orthogonal connectors for normal feature flow. Dashed coral connectors denote optional/callout relations; dashed violet denotes semantic guidance.
- Arial or Helvetica throughout. Hierarchy: section labels > module labels > tensor dimensions.
- Light tinted containers group Backbone, ASPP, Decoder, TOFAA V2, and TOFAA V3.
- Minimal legend at lower right: feature flow, optional/guidance flow, and feature map.

## Quality Checks

- Verify all labels and tensor sizes against `generator.md`.
- Verify Layer 1 supplies the low-level path and Layer 4 supplies ASPP.
- Verify ASPP reaches both the decoder and the V3 semantic-guidance inset.
- Verify all arrowheads point in the direction of computation.
- Check alignment, spacing, overlaps, clipping, typography, and color consistency at export size.
- Validate the editable source structurally before delivery.

