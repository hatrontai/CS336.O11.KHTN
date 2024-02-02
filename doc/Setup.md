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
3. Install [redis-stack](https://redis.io/docs/install/install-stack/) which will be used in index construction
# Setup
Download [Flickr30k dataset](https://www.kaggle.com/datasets/adityajn105/flickr30k) and extract it in the repository with the name "flickr30k"

Construct index
```bash
python construct_index.py
```
Start system with user interface
```bash
python main.py
```