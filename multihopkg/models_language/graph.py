import torch
from torch import nn
import pdb
import math
from typing import List, Optional
import os

from transformers import BartForConditionalGeneration, BartTokenizer, PreTrainedTokenizer

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
        self.graphbart_encoder_layers = [GraphEncoderLayer(answer_tokenizer, graph_embedding_dim) for _ in range(num_layers)]
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

class GraphMultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(GraphMultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = model_dim
        self.num_heads = num_heads
        self.d_k = model_dim // num_heads

        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.W_o = nn.Linear(model_dim, model_dim)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output      return x       return x
