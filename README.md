# DMX-Speculative-Decoding

## Speculative Decoding Framework

A generalizable approach to accelerating AI inference across multiple modalities (text-to-video, text-to-audio, text-to-speech) by leveraging speculative decoding techniques for faster and more efficient generation.

## Overview

Accelerating AI inference for multimodal LLMs by exploring generalizable speculative decoding techniques. By combining lightweight draft models with high-quality refinement models and employing selective verification mechanisms, this approach achieves 2.5x to 6x speedups while maintaining output quality.

## Features

- **Multi-modal support**: Works with text-to-video, text-to-audio, and text-to-speech generation
- **Advanced quantization**: Uses TorchAO int8 weight-only quantization to reduce memory footprint
- **Selected verification mechanisms**: Custom metrics for each modality to ensure high-quality outputs

## Installation

## Clone the repository
```python
git clone https://github.com/trirpi/Hackaton-DMX.git
```

## Navigate to the project directory
```python
cd Hackaton-DMX
```

## Install the package in development mode
```python
pip install -e .
```

# Usage

## Text-to-Video Generation

Generate high-quality videos with significantly faster inference:

```python dmx_speculative_decoding/text_to_video.py```

## Customizing Generation
You can customize the generation by modifying parameters:

### Example parameters
```
prompt = "Your custom prompt here"
height = 512
width = 512
num_frames = 16
main_steps = 2
draft_steps = 2
refinement_threshold = 0.3
```

### How It Works

**Draft Generation**: A lightweight model quickly generates initial content
   
**Quality Analysis**: Effective metrics identify which parts need refinement

**Selective Refinement**: Only critical parts are processed by the high-quality model

**Quantization**: Models are quantized to reduce memory usage and improve performance

**Text-to-Video**: Up to 6x faster at 512x512 resolution
**Text-to-Audio**: 3x speedup with comparable quality
**Text-to-Speech**: 2.5x faster inference while preserving natural speech

## Requirements  

- Nvidia A100 GPU (recommended)  
- See `pyproject.toml` for full dependencies 


