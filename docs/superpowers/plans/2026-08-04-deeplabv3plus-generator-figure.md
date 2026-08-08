# DeepLabV3+ Generator Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and export an editable, publication-ready 2D DeepLabV3+ generator architecture figure with compact TOFAA V2/V3 insets.

**Architecture:** Construct one 16:9 draw.io page from semantic vector cells. The upper region contains the main left-to-right DeepLabV3+ dataflow, while the lower region contains two simplified attention-module insets connected by dashed callouts.

**Tech Stack:** draw.io MCP, mxGraph vector geometry, PNG/SVG export, PowerShell read-only verification.

## Global Constraints

- Use English-only labels.
- Use a flat 2D layout on a white background.
- Use muted blue for Backbone, violet for ASPP, coral for TOFAA, sage green for Decoder, and graphite/light gray for Input/Output.
- Use orthogonal feature-flow connectors with explicit direction and no ambiguous crossings.
- Provide editable `.drawio` plus approximately 2000 px review PNG; also provide SVG when available.
- Preserve the tensor sizes and topology defined in `generator.md`.

---

### Task 1: Establish the Draw.io Canvas and Visual System

**Files:**
- Reference: `generator.md`
- Reference: `2D绘图skill.md`
- Create: `DeepLabV3Plus_Generator_TOFAA.drawio`

**Interfaces:**
- Consumes: the approved design specification.
- Produces: one draw.io page named `DeepLabV3+ Generator`, with reusable cell styles and stable semantic IDs.

- [ ] **Step 1: Start the live draw.io session**

Call `start_session`, then create a new one-page diagram with `<mxGraphModel pageWidth="1600" pageHeight="900" background="#FFFFFF">` and root cells `0` and `1`.

- [ ] **Step 2: Add group containers and headings**

Create non-swimlane background containers with stable IDs `backbone_group`, `aspp_group`, `decoder_group`, `tofaa_v2_group`, and `tofaa_v3_group`. Use low-opacity fills, dark-gray borders, and section headings inside each container.

- [ ] **Step 3: Verify the canvas structure**

Fetch the diagram and confirm there is exactly one page, all five containers exist, and all content coordinates fit within `0 <= x <= 1600` and `0 <= y <= 900`.

### Task 2: Build the DeepLabV3+ Main Pipeline

**Files:**
- Modify: `DeepLabV3Plus_Generator_TOFAA.drawio`

**Interfaces:**
- Consumes: the visual system and container IDs from Task 1.
- Produces: stable nodes `input`, `stem`, `layer1`, `layer2`, `layer3`, `layer4`, ASPP branch nodes, `aspp_fuse`, `low_proj`, `upsample_high`, `concat`, decoder nodes, and `output`.

- [ ] **Step 1: Add input, backbone stages, and feature flow**

Add the input feature map and five backbone stage blocks. Labels must include `Depth + IR`, `Stem`, `Layer 1` through `Layer 4`, bottleneck counts, and the exact tensor dimensions `2 x 576 x 640`, `256 x 144 x 160`, `512 x 72 x 80`, `1024 x 36 x 40`, and `2048 x 18 x 20`.

- [ ] **Step 2: Add the five-branch ASPP**

Create branch nodes labelled `1x1 Conv`, `3x3 Atrous r=6`, `3x3 Atrous r=12`, `3x3 Atrous r=18`, and `Image Pooling`. Route each from Layer 4 and merge through `Concat + 1x1 Conv` to `256 x 18 x 20`.

- [ ] **Step 3: Add low-level and decoder paths**

Route Layer 1 through `Optional TOFAA` and `1x1 Conv 256->48`. Route ASPP through `Bilinear Upsample`. Merge both at `Concat (304 ch)`, then add `3x3 Conv`, `3x3 Conv`, `1x1 Classifier`, and `4x Bilinear Upsample` before the output node.

- [ ] **Step 4: Verify main topology**

Fetch the diagram XML and confirm all required node IDs exist; confirm `layer1` reaches `low_proj`, `layer4` reaches every ASPP branch, `aspp_fuse` reaches the decoder, and the final edge terminates at `output`.

### Task 3: Build the Simplified TOFAA V2/V3 Insets

**Files:**
- Modify: `DeepLabV3Plus_Generator_TOFAA.drawio`

**Interfaces:**
- Consumes: `optional_tofaa` and `aspp_fuse` nodes from Task 2.
- Produces: compact insets communicating the V2 physical gate and V3 physical-semantic joint modulation.

- [ ] **Step 1: Add TOFAA V2**

Add the sequence `Depth / IR Encoding -> Physical Gate -> Channel + Spatial Attention -> Feature Fusion -> Residual Gate`, with a residual bypass from the low-level input to an addition node.

- [ ] **Step 2: Add TOFAA V3**

Add `Depth / IR Encoding` and `ASPP Semantic Guidance` feeding `Physical-Semantic Joint Modulation`, followed by `Channel + Spatial Attention -> Feature Fusion -> Residual Gate`, with the same residual bypass.

- [ ] **Step 3: Add callouts and legend**

Connect the main `Optional TOFAA` node to both inset containers with dashed coral callouts. Connect `aspp_fuse` to `ASPP Semantic Guidance` with a dashed violet guidance line. Add a compact legend for solid feature flow, dashed optional/guidance flow, and feature-map rectangles.

- [ ] **Step 4: Verify inset meaning and routing**

Confirm no dashed callout is mistaken for normal feature flow, V3 alone receives ASPP semantic guidance, and all connectors avoid unrelated nodes.

### Task 4: Export and Quality-Assure Deliverables

**Files:**
- Create: `DeepLabV3Plus_Generator_TOFAA.drawio`
- Create: `DeepLabV3Plus_Generator_TOFAA.png`
- Create: `DeepLabV3Plus_Generator_TOFAA.svg`

**Interfaces:**
- Consumes: completed draw.io page.
- Produces: editable source and manuscript-ready raster/vector exports.

- [ ] **Step 1: Export all formats**

Use `export_diagram` to save `.drawio`, `.png`, and `.svg` files in the workspace root.

- [ ] **Step 2: Inspect the rendered figure**

Open the PNG and check typography, clipping, alignment, whitespace, colors, arrowheads, connector overlaps, and visual hierarchy.

- [ ] **Step 3: Validate source and outputs**

Verify that the source contains all stable semantic IDs, the PNG and SVG exist and are non-empty, and the exported pixel ratio is suitable for the approved wide composition.

- [ ] **Step 4: Correct defects and re-export**

If inspection finds a defect, update only the affected draw.io cells, re-export all formats, and repeat Steps 2--3 until the checks pass.

