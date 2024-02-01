from lavis.models import load_model_and_preprocess
import torch
from transformers import BertTokenizer
import torch.nn.functional as F

def get_embed_dim():
    return 256

def init_model(model_size): #model_size must be "base" or "large"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name = "blip_feature_extractor", 
                                                                          model_type = model_size, 
                                                                          is_eval = True, 
                                                                          device = device)
    return model, vis_processors, txt_processors

def init_tokenizer(): 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

def image_encoder(image, model, vis_processors, device):
    image_processed = vis_processors["eval"](image).unsqueeze(0).to(device)
    image_embeds = model.visual_encoder.forward_features(image)
    image_features = model.vision_proj(image_embeds)
    image_features = F.normalize(image_features, dim=-1)
  
    embedding = image_features[0].detach().numpy() # get embedding of cls tokens on ViT for representation vector.
    return embedding

def text_encoder(text, model, tokenizer, text_processors, device):
    text_input = txt_processors["eval"](text)
    text = tokenizer(text_input, return_tensors="pt", padding=True).to(device)
    text_output = model.text_encoder(
                text.input_ids,
                attention_mask = text.attention_mask,
                return_dict = True,
                mode = "text",
            )
    text_embeds = text_output.last_hidden_state
    text_features = model.text_proj(text_embeds)
    text_features = F.normalize(text_features, dim=-1)
    embedding = text_features[0][0].detach().numpy() # get embedding of cls tokens on BERT for representation vector.
    return embedding
