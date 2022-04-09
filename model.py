import torch
from torch import nn
import torch.nn.functional as F

class PrefrozenEmbeddings(nn.Module):
    def __init__(self, frozen_embeddings: nn.Embedding, num_extra_embeddings: int):
        super().__init__()
        self.frozen = frozen_embeddings
        self.frozen.weight.requires_grad = False

        self.raw = nn.Embedding(
            num_extra_embeddings, self.frozen.embedding_dim,
            padding_idx=self.frozen.padding_idx, max_norm=self.frozen.max_norm,
        )

    def forward(self, input):
        frozen_weights = self.frozen.weight
        raw_weights = self.raw.weight
        weights = torch.cat([frozen_weights, raw_weights], axis=0)
        return F.embedding(
            input, weights, self.frozen.padding_idx, self.frozen.max_norm,
            self.frozen.norm_type, self.frozen.scale_grad_by_freq, self.frozen.sparse
        )


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_sentences(sentences, model, tokenizer, device):
    input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model(**input)
    sentence_embeddings = mean_pooling(outputs, input['attention_mask'])
    return sentence_embeddings