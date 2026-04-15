# Installation

## Requirements

## Setup 
First, create conda environment.
```
conda create --name vatex python=3.9 -y
conda activate vatex
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

```
Then, clone the repository and install the necessary packages.
```
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

pip install -r requirements.txt

# Install spacy
python -m spacy download en_core_web_sm

# CUDA operators for MS Deformable Attn Decoder
cd vatex/modeling/pixel_decoder/ops
sh make.sh
```
