import numpy as np
import torch
import redis
import os
from PIL import Image


from blip import get_embed_dim, blip
from redis.commands.search.field import (
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from tqdm import tqdm 
from time import time

class database_manager:
    def __init__(self, use_cpu = False):
        self.client = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # Define index params
        self.index_name = 'index10'
        self.doc_prefix = 'img:'

        # Image path to retrieve
        self.img_path = "flickr30k-images"

        # Initialize model
        self.model = blip(use_cpu = use_cpu)
        self.query_total = 0.0
        self.encode_total = 0.0

    def create_index(self, distance_metric = "COSINE", counter = False):

        try:
        # check to see if index exists
            print("Checking if index existed...")
            self.client.ft(self.index_name).info()
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
                        "DISTANCE_METRIC":distance_metric,
                    },
                ),
            )

            # index Definition
            definition = IndexDefinition(prefix=[self.doc_prefix], index_type=IndexType.HASH)

            # create Index
            self.client.ft(self.index_name).create_index(fields=schema, definition=definition)
            
            pipe = self.client.pipeline()

            cnt = 0
            for name in tqdm(os.listdir(self.img_path)):
                if name[-3:] == 'jpg':
                    obj = {}
                    
                    pipe = self.client.pipeline()
                    
                    img_dir = os.path.join(self.img_path, name)
                    img = Image.open(img_dir)
                    
                    #objects.append({"dir":os.path.join(img_path,name), "embedding":embedding})
                    obj['dir'] = img_dir

                    embedding = self.model.encode_image(img)
                    obj['embedding'] = embedding.astype(np.float32).tobytes()
                    key = f'img:{name[:-4]}'

                    if counter:
                        cnt = cnt+1
                        print(cnt)

                    pipe.hset(key, mapping = obj)
                    res = pipe.execute()
                    #print(f'error: {res}')
                    


    def query(self, search_text, topk = 30, enable_time_log = False):
        encode_time = time()
        embedding = self.model.encode_text(search_text)

        query_time = time()
        query = (
            Query(f"*=>[KNN {topk} @embedding $vec as score]")
             .sort_by("score")
             .return_fields("dir")
             .paging(0, topk)
             .dialect(2)
        )
        #print(embedding)
        query_params = {
            "vec": embedding.astype(np.float32).tobytes()
        }
        result = self.client.ft(self.index_name).search(query, query_params).docs
        end_time = time()

        if enable_time_log == True:
            self.query_total += end_time - query_time
            self.encode_total += query_time - encode_time
        return result


