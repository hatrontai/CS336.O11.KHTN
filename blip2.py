from lavis.models import load_model_and_preprocess
import torch
from transformers import BertTokenizer
import torch.nn.functional as F

def get_embed_dim():
    return 256

def init_model(model_size = 'pretrain'): #model_size must be "pretrain"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name = "blip2_image_text_matching", 
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
    image_processed = vis_processors["eval"](img).unsqueeze(0).to(device)
    with model.maybe_autocast():
        image_embeds = model.ln_vision(model.visual_encoder(image_processed))
    image_embeds = image_embeds.float()
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )
    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)

    query_output = model.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
    image_feats = F.normalize(model.vision_proj(query_output.last_hidden_state), dim=-1)
  
    return image_feats[0][0].detach().cpu().numpy()

def text_encoder(text, model, tokenizer, text_processors, device):
    text_input = text_processors["eval"](text)
    text = tokenizer(text_input, return_tensors="pt", padding=True).to(device)
    text_output = model.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
    text_feat = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )
    return text_feat[0].detach().cpu().numpy()
