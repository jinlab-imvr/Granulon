
# Granulon: Awakening Pixel-Level Visual Encoders with Adaptive Multi-Granularity Semantics for MLLM


<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2602.14512-b31b1b.svg)](https://arxiv.org/abs/2603.08800)&nbsp;
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2603.08800)&nbsp;

</div>

<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2603.08800">
    Granulon: Awakening Pixel-Level Visual Encoders with Adaptive Multi-Granularity Semantics for MLLM
  </a>
</p>

<div>
  <p align="center" style="font-size: larger;">
    <strong>Awakening the potential of pixel-level visual encoders for MLLMs through adaptive multi-granularity semantic modeling</strong>
  </p>
</div>


<br>
---

## Overview

<!-- HERO FIGURE -->
![Granulon Overview](figures/overview.png)

Multimodal Large Language Models (MLLMs) have achieved remarkable success by leveraging CLIP-based visual encoders that provide strong global semantic alignment. However, these models often struggle with **fine-grained visual understanding** and pixel-level perception.

On the other hand, modern self-supervised visual encoders such as **DINOv3** exhibit powerful **pixel-level representation capability**, but they lack coarse-grained semantic abstraction that aligns well with language.

**Granulon** bridges this gap by introducing **adaptive multi-granularity semantics**, enabling pixel-level visual encoders to dynamically adjust their semantic abstraction levels according to the textual context.

---

## Motivation

<!-- MOTIVATION FIGURE -->
![Motivation](figures/motivation.png)

Existing multimodal visual encoders face a fundamental trade-off:

| Encoder Type | Strength | Limitation |
|---------------|----------|------------|
| CLIP-style encoders | Strong global semantic alignment | Weak pixel-level understanding |
| Self-supervised encoders (e.g., DINO) | Rich pixel-level features | Lack semantic abstraction |

MLLM reasoning often requires **multiple levels of visual granularity**, including:

- object-level semantics
- region-level relations
- fine-grained pixel perception

Granulon addresses this by enabling **adaptive visual granularity conditioned on text semantics**.

---

## Method Overview

<!-- METHOD PIPELINE -->
![Method Pipeline](figures/pipeline.png)

Granulon introduces a framework that augments pixel-level visual encoders with **adaptive semantic abstraction mechanisms**.

The framework mainly consists of two key components:

1. **Granularity Controller**
2. **Adaptive Token Aggregation**

Together, these modules dynamically organize visual tokens into semantically meaningful structures that align with language reasoning.

---

## Granularity Controller

<!-- CONTROLLER FIGURE -->
![Granularity Controller](figures/controller.png)

The **Granularity Controller** dynamically determines the level of visual abstraction required for a given textual query.

Instead of using a fixed representation level, the controller:

- analyzes the **semantic scope of the input text**
- predicts the appropriate **visual granularity**
- guides the token aggregation process

This mechanism allows the model to flexibly switch between:

- **fine-grained perception**
- **region-level reasoning**
- **coarse semantic abstraction**

---

## Adaptive Token Aggregation

<!-- TOKEN AGGREGATION -->
![Token Aggregation](figures/token_aggregation.png)

To generate compact and semantically rich visual representations, Granulon introduces **Adaptive Token Aggregation**.

This module performs:

- **granularity-aware pooling**
- **relation-aware clustering**

Key ideas include:

- grouping visual tokens according to semantic relevance
- preserving spatial relationships
- maintaining fine-grained details when necessary

The resulting tokens provide a **hierarchical representation** suitable for multimodal reasoning.

---

## Multi-Granularity Visual Semantics

<!-- MULTI GRANULARITY -->
![Multi Granularity](figures/multigranularity.png)

Granulon enables visual representations to span multiple semantic levels:

- **Pixel-level**
- **Region-level**
- **Object-level**
- **Scene-level**

By dynamically adapting the representation granularity, the model improves:

- fine-grained visual reasoning
- cross-modal alignment
- semantic abstraction for language interaction

---

## Experimental Insights

<!-- RESULTS FIGURE -->
![Results](figures/results.png)

Granulon demonstrates strong performance across multimodal reasoning tasks.

Key observations include:

- Pixel-level encoders can be effectively awakened for multimodal reasoning.
- Adaptive granularity improves semantic alignment between visual tokens and text.
- Multi-granularity representations enhance both perception and reasoning capabilities.

---

## Visualization

<!-- VISUALIZATION -->
![Visualization](figures/visualization.png)

Visualization results show how Granulon dynamically adapts visual granularity depending on textual queries.

Examples illustrate:

- fine-grained object localization
- semantic region grouping
- hierarchical token organization

---

## Discussion

Granulon highlights the importance of **adaptive semantic abstraction** for multimodal perception.

Rather than relying solely on globally aligned encoders such as CLIP, enabling pixel-level encoders to **adaptively organize visual information** provides a promising direction for future multimodal models.

This paradigm suggests that **multi-granularity representation learning** may be a key ingredient for next-generation MLLMs.

---

## Citation

If you find this work useful, please consider citing:
