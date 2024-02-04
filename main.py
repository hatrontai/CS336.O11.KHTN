from database_manager import database_manager
import redis
import gradio

import numpy as np
import gradio as gr
from database_manager import database_manager
from PIL import Image
import PIL

manager = database_manager()

def Processing(text_input):
    rankedlist_retrieved = manager.query(text_input, topk = 30)
    
    imgs_retrieved = []
    for doc in rankedlist_retrieved:
        dir_image = doc['dir']
        image = Image.open(dir_image)
        imgs_retrieved.append(image)
    return imgs_retrieved
    

demo = gr.Interface(
    fn= Processing, 
    inputs= "text",
    outputs= gr.Gallery(
        label="Image retrieved", type = 'pil', columns = 5),
    title = 'Image Retrieval',
    description = 'Group 6'
    )

if __name__ == "__main__":
    demo.launch(share= True)
