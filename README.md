# DMX-Speculative-Decoding

Speculative Decoding Framework
A generalized framework for accelerating AI generation across multiple modalities (text-to-video, text-to-audio, text-to-speech) using speculative decoding techniques.
Overview

This project implements a novel approach to accelerate generative AI models by combining lightweight draft models with high-quality refinement models. By using speculative decoding techniques and sophisticated verification methods, we achieve significant speedups (2.5x to 7x) while maintaining comparable output quality.
Features

Multi-modal support: Works with text-to-video, text-to-audio, and text-to-speech generation

Adaptive threshold systems: Dynamically determines which content needs refinement

Advanced quantization: Uses TorchAO int8 weight-only quantization to reduce memory footprint
Sophisticated quality metrics: Custom metrics for each modality to ensure high-quality outputs

# Installation

## Clone the repository
git clone https://github.com/trirpi/Hackaton-DMX.git

## Navigate to the project directory
cd Hackaton-DMX

## Install the package in development mode
```python
pip install -e .
```

# Usage

## Text-to-Video Generation

Generate high-quality videos with significantly faster inference:

```python dmx_speculative_decoding/text_to_video.py```

## Customizing Generation
You can customize the generation by modifying parameters in the script:

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
   
**Quality Analysis**: Sophisticated metrics identify which parts need refinement

**Selective Refinement**: Only critical parts are processed by the high-quality model

**Quantization**: Models are quantized to reduce memory usage and increase speed Performance

**Text-to-Video**: Up to 6x faster at 512x512 resolution
**Text-to-Audio**: 3x speedup with comparable quality
**Text-to-Speech**: 2.5x faster inference while preserving natural speech

## Requirements
```
Python 3.8+
PyTorch 2.0+
```
CUDA-capable GPU (recommended)
See pyproject.toml for full dependencies

## Citation
If you use this framework in your research, please cite our work:
```
@software{speculative_decoding_framework,
  author = {Hackaton-DMX Team},
  title = {Dmx Speculative Decoding Framework},
  year = {2025},
  url = {https://github.com/trirpi/Hackaton-DMX}
}
```

### License
MIT

