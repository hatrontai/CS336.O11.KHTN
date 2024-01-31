import redis
from tmp import text_encoder, image_encoder, get_embed_dim
import numpy as np
import redis
from redis.commands.search.field import (
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import os
from PIL import Image


class database_manager:
    def __init__(self):
        self.client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.index_name = 'index1'
        self.doc_prefix = 'img:'
        self.img_path = "flickr30k/Images"

    def create_index(self, distance_metric = "COSINE", counter = False):

        try:
        # check to see if index exists
            self.client.ft(self.index_name).info()
            print("Index already exists!")        

        
        except:
            embedding_dim = get_embed_dim()
            print(embedding_dim)
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
            for name in os.listdir(self.img_path):
                if name[-3:] == 'jpg':
                    obj = {}
                    
                    img_dir = os.path.join(self.img_path, name)
                    img = Image.open(img_dir)
                    
                    #objects.append({"dir":os.path.join(img_path,name), "embedding":embedding})
                    obj['dir'] = img_dir

                    embedding = image_encoder(img)
                    
                    obj['embedding'] = embedding.astype(np.float32).tobytes()
                    key = f'img:{name[:-4]}'

                    if counter:
                        cnt = cnt+1
                        print(cnt)

                    pipe.hset(key, mapping = obj)
            res = pipe.execute()
                    


    def query(self, search_text, topk = 40):
        embedding = text_encoder(search_text)
        # Search using embedding
        # Return ranked list of topk
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
        return result


