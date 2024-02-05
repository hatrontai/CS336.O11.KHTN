# Installation
1. (Optional) Creating conda environment
```bash
conda create -n cs336 python=3.8
conda activate cs336
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Install [Redis Stack](https://redis.io/docs/install/install-stack/) which will be used in index construction
# Setup
1. Download [Flickr 30k images](https://shannon.cs.illinois.edu/DenotationGraph/) and extract it in the repository with the name "flickr30k-images"

2. Start Redis Stack

3. Construct index from scratch or load precalculated embeddings
- Construct index
```bash
python construct_index.py
```
- Download [embedding file](https://www.kaggle.com/datasets/iambestfeeder/annotations-flickr30k?select=image_features_blip_feature_extractor_base.json) to `/data/` and load embeddings
```bash
python load_index.py
```
4. Start system with user interface
```bash
python main.py
```