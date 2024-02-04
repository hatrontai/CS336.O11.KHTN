from lavis.models import load_model_and_preprocess
import torch
from transformers import BertTokenizer
import torch.nn.functional as F


    

def get_embed_dim():
    return 256

def init_tokenizer(): 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

class blip:
    def __init__(self, model_size = 'base', use_cpu = False): # model_size must be "base" or "large"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (use_cpu):
            self.device = 'cpu'
        
        self.model, self.vis_processor, self.txt_processor = load_model_and_preprocess(name = "blip_feature_extractor", 
                                                                          model_type = model_size, 
                                                                          is_eval = True, 
                                                                          device = self.device)
        self.tokenizer = init_tokenizer()
        

    def encode_image(self, image):
        image_processed = self.vis_processor["eval"](image).unsqueeze(0).to(self.device)
        image_embeds = self.model.visual_encoder.forward_features(image_processed)
        image_features = self.model.vision_proj(image_embeds)
        image_features = F.normalize(image_features, dim=-1)
      
        embedding = image_features[0][0].detach().cpu().numpy() # get embedding of cls tokens on ViT for representation vector.
        return embedding

    def encode_text(self, text):
        text_input = self.txt_processor["eval"](text)
        text = self.tokenizer(text_input, return_tensors="pt", padding=True).to(self.device)
        text_output = self.model.text_encoder(
                    text.input_ids,
                    attention_mask = text.attention_mask,
                    return_dict = True,
                    mode = "text",
                )
        text_embeds = text_output.last_hidden_state
        text_features = self.model.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)
        embedding = text_features[0][0].detach().cpu().numpy() # get embedding of cls tokens on BERT for representation vector.
        return embedding

