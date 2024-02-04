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

class blip2:
    def __init__(self, model_size='pretrain'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processor, self.text_processor = load_model_and_preprocess(name = "blip2_image_text_matching", 
                                                                          model_type = model_size, 
                                                                          is_eval = True, 
                                                                          device = self.device)
        self.tokenizer = init_tokenizer()
        self.model = self.model.to(torch.float)

    def encode_image(self, image):
        image_processed = self.vis_processor["eval"](image).unsqueeze(0).to(torch.float).to(self.device)
        
        image_embeds = self.model.ln_vision(self.model.visual_encoder(image_processed))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image_embeds.device
                )
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.model.Qformer.bert(
                            query_embeds=query_tokens,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )
        image_feats = F.normalize(self.model.vision_proj(query_output.last_hidden_state), dim=-1)
      
        return image_feats[0][0].detach().cpu().numpy()


    def encode_text(self, text):
        text_input = self.text_processor["eval"](text)
        text = self.tokenizer(text_input, return_tensors="pt", padding=True).to(self.device)
        text_output = self.model.Qformer.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                )
        text_feat = F.normalize(
                    self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
                )
        return text_feat[0].detach().cpu().numpy()
