import torch
from torch import nn
import pdb
import math
from typing import List, Optional
import os

class GraphBart(nn.Module):
    """
    Bart with Decoder as Graph Attention.
    Meant to be trained on embeddings from graph space and output a reconstruction in natural language.
    """

    def __init__(
        self,
        pretrained_bart_model: str,
        answer_tokenizer: PreTrainedTokenizer,
        graph_embedding_dim: int,
    ):
        super(GraphBart, self).__init__()
        # TODO: ENsure this is correct
        self.graphbart_encoder_layers = [GraphBartEncoderLayer(answer_tokenizer, graph_embedding_dim) for _ in range(num_layers)]
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_bart_model)
        self.bart_hidden_dim = self.bart.config.d_model
        self.pretrained_bart_tokenizer = answer_tokenizer

    def forward(
        self,
        graph_embeddings: torch.Tensor,
        graph_mask: torch.Tensor, # Adjacency Matrix
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels=None,
    ):
        # Pass graph embeddings through custom encoder
        # Pass the outputs to BART decoder
        # translated_embeddings = graph_embeddings

        outputs = self.bart(
            inputs_embeds=translated_embeddings,
            decoder_input_ids=decoder_input_ids, #For teacher forcing. 
        )
        return outputs

class GraphEncoderLayer(nn.Module):
   def __init__(self, d_model, num_heads, d_ff, dropout):
       super(GraphEncoderLayer, self).__init__()
       self.self_attn = MultiHeadAttention(d_model, num_heads)
       self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
       self.norm1 = nn.LayerNorm(d_model)
       self.norm2 = nn.LayerNorm(d_model)
       self.dropout = nn.Dropout(dropout)

   def forward(self, x, mask):
       attn_output = self.self_attn(x, x, x, mask)
       x = self.norm1(x + self.dropout(attn_output))
       ff_output = self.feed_forward(x)
       x = self.norm2(x + self.dropout(ff_output))
       return x
