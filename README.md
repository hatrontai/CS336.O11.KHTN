# CS336.O11.KHTN - Final Project
## Information
**Instructor:** Ph.D Ngo Duc Thanh

**Team members:**
- Nguyen Duc Nhan - 21520373
- Le Chau Anh - 21521821
- Vo Minh Quan - 21520093
- Ha Trong Tai - 21520436
- Nguyen Nhat Minh - 21521135

**Year:** 2023-2024
## Image retrieval system
### Installation
1. (Optional) Creating conda environment
```bash
conda create -n cs336 python=3.8
conda activate cs336
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Install [Redis Stack](https://redis.io/docs/install/install-stack/) which will be used as vector database in our retrieval system.
### Setup
1. Download [Flickr 30k images](https://shannon.cs.illinois.edu/DenotationGraph/) and extract it in the repository with the name "flickr30k-images"

2. Start Redis Stack

3. Construct index from scratch or load precalculated embeddings
- Construct index
```bash
python construct_index.py
```
- Download [embedding file](https://www.kaggle.com/datasets/iambestfeeder/annotations-flickr30k?select=image_features_blip_feature_extractor_base.json) to `data/` and load embeddings
```bash
python load_index.py
```
4. Start system with user interface
```bash
python main.py
```
### Acknowledge
This system is deployed thanks to:
- [lavis](https://github.com/salesforce/LAVIS/tree/5ddd9b4e5149dbc514e81110e03d28458a754c5d)
- [torch](https://pytorch.org)
- [transformers](https://github.com/huggingface/transformers)
- [redis](https://redis.io/)
- [gradio](https://www.gradio.app/)

