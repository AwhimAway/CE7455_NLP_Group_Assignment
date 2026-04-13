"""

Model code for TinyStyler.

TinyStyler wraps a T5 model, and preprends a style embedding to the input. 

"""

import torch
from transformers import T5ForConditionalGeneration


# For model specific logic
MODEL_TO_MODEL_TYPE = {
    'google/t5-v1_1-large': 't5',
    't5-large': 't5',
    't5-small': 't5-small',
    't5-base' : 't5-base',
    'google/t5-efficient-base': 'google/t5-efficient-base',
}


class TinyStyler(torch.nn.Module):
    def __init__(self, base_model, use_style=False, ctrl_embed_dim=768):
        super().__init__()

        if MODEL_TO_MODEL_TYPE[base_model] == 't5' or 't5-small' or 't5-base' or 'google/t5-efficient-base':
            self.model = T5ForConditionalGeneration.from_pretrained(base_model)
        else:
            assert False
        self.use_style = use_style
        if self.use_style:
            self.ctrl_embed_dim = ctrl_embed_dim
            if hasattr(self.model.config, 'd_model'):
                self.proj = torch.nn.Linear(
                    self.ctrl_embed_dim, self.model.config.d_model
                )
            else:
                self.proj = torch.nn.Linear(
                    self.ctrl_embed_dim, self.model.config.hidden_size
                )
        print("Loaded base model:", base_model)
        print("Model config:", self.model.config)
        print("d_model:", self.model.config.d_model)
        print("num_layers:", self.model.config.num_layers)

    def forward(self, input_ids, attention_mask, labels=None, style=None):
        if self.use_style:
            style_embed = self.proj(style).unsqueeze(1)

        input_embeds = self.model.get_input_embeddings()(input_ids)
        if self.use_style:
            input_embeds = torch.cat([style_embed, input_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones((input_embeds.shape[0], 1)).to(attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )

        return self.model(
            inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels
        )

    def generate(self, input_ids, attention_mask, style=None, **kwargs):
        if self.use_style:
            style_embed = self.proj(style.unsqueeze(1))

        input_embeds = self.model.get_input_embeddings()(input_ids)
        if self.use_style:
            input_embeds = torch.cat([style_embed, input_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones((input_embeds.shape[0], 1)).to(attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )

        return self.model.generate(
            inputs_embeds=input_embeds, attention_mask=attention_mask, **kwargs
        )
