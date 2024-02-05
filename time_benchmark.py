from database_manager import database_manager
import redis
import gradio

import numpy as np
import json
from database_manager import database_manager
from PIL import Image
import PIL
from time import time
import pandas as pd

manager = database_manager(use_cpu = True)

testset = pd.read_json('data/testset.json')
n = len(testset)

for i in range(n):
    res = manager.query(testset.iloc[i][' comment'], enable_time_log = True)

print(f'Average encode time is {manager.encode_total/(1.0*n)}')
print(f'Average query time is {manager.query_total/(1.0*n)}')
