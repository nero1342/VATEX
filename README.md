The official implementation of the paper:

<div align="center">
<h1>
<b>
VATEX: Improving Referring Image Segmentation <br> using Vision-Aware Text Features
</b>
</h1>
</div>
<p align="center">
  <img src="assets/Overview.png" width="600">
</p>

## Abstract
Referring image segmentation is a challenging task that involves generating pixel-wise segmentation masks based on natural language descriptions. Existing methods have relied mostly on visual features to generate the segmentation masks while treating text features as supporting components. This over-reliance on visual features can lead to suboptimal results, especially in complex scenarios where text prompts are ambiguous or context-dependent. To overcome these challenges, we present a novel framework VATEX to improve referring image segmentation by enhancing object and context understanding with Vision-Aware Text Feature. Our method involves using CLIP to derive a CLIP Prior that integrates an object-centric visual heatmap with text description, which can be used as the initial query in DETR-based architecture for the segmentation task. Furthermore, by observing that there are multiple ways to describe an instance in an image, we enforce feature similarity between text variations referring to the same visual input by two components: a novel Contextual Multimodal Decoder that turns text embeddings into vision-aware text features, and a Meaning Consistency Constraint to ensure further the coherent and consistent interpretation of language expressions with the context understanding obtained from the image. Our method achieves a significant performance improvement on three benchmark datasets RefCOCO, RefCOCO+ and G-Ref. 

## Update
``` update here ```

## Demo
<p align="center">
  <img src="assets/demo.gif" width="600">
</p>

## Main Results

Main results on RefCOCO

| Backbone | val | test A | test B |
| ---- |:-------------:| :-----:|:-----:|
| ResNet50 | 73.92 | 76.03 | 70.86 |
| ResNet101 | 74.67  |   76.8  | 70.42 |

Main results on RefCOCO+

| Backbone | val | test A | test B |
| ---- |:-------------:| :-----:|:-----:|
| ResNet50 | 64.02 | 69.74 | 55.04 |
| ResNet101 | 64.80 | 70.33 | 56.33 |

Main results on G-Ref

| Backbone | val | test |
| ---- |:-------------:| :-----:|
| ResNet50 | 65.69 | 65.90 |
| ResNet101 | 66.77 | 66.52|

## Requirements
We test our work in the following environments, other versions may also be compatible:
- CUDA 11.1
- Python 3.8
- Pytorch 1.9.0

## Installation
Please refer to [installation.md](docs/installation.md) for installation

## Data preparation
Please refer to [data.md](docs/data.md) for data preparation.

## Training 

## Evaluation
