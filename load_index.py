import pandas as pd
import numpy as np
import torch
import redis
import os

from blip import get_embed_dim
from redis.commands.search.field import (
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from tqdm import tqdm 
from time import time


image_features = pd.read_json('data/image_features_blip_feature_extractor_base.json', lines = True)
image_embeddings = np.array(image_features['img_emb'].to_list())

index_name = 'index10'
doc_prefix = 'img:'

img_path = 'flickr30k-images'

client = redis.Redis()




try:
    # check to see if index exists
    print("Checking if index existed...")
    client.ft(index_name).info()
    print("Index already exists!")        

except:
    print("Constructing index...")
    embedding_dim = get_embed_dim()
    schema = (
        TextField("dir"),
        VectorField(
            "embedding",
            "FLAT",
            {
                "TYPE":"FLOAT32",
                "DIM":embedding_dim,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )
    # index Definition
    definition = IndexDefinition(prefix=[doc_prefix], index_type=IndexType.HASH)

    client.ft(index_name).create_index(fields=schema, definition=definition)

    
    for i in tqdm(range(len(image_features))):
        pipe = client.pipeline()
        name = image_features.iloc[i]['image_name']
        obj = {}

        obj['dir'] = os.path.join(img_path, name) 
        obj['embedding'] = image_embeddings[i].astype(np.float32).tobytes()

        key = f'img:{name[:-4]}'

        pipe.hset(key, mapping = obj)
        pipe.execute()
                   
