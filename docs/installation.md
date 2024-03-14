# Installation

## Requirements

## Setup 
First, create conda environment.
```
conda create --name vatex python=3.8 -y
conda activate vatex
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
```
Then, clone the repository and install the necessary packages.
```
pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install -r requirements.txt

# Install spacy
python -m spacy download en_core_web_sm

# CUDA operators for MS Deformable Attn Decoder
cd vlformer/modeling/pixel_decoder/ops
sh make.sh
```
