import torch
from torch import nn
from transformers import BartForConditionalGeneration, BartTokenizer

import rich
from rich import traceback

traceback.install()

# Step 1: Define the custom encoder
class CustomGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomGraphEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, graph_embeddings):
        x = self.activation(self.fc1(graph_embeddings))
        return self.fc2(x)

# Step 2: Load pretrained BART and replace the encoder
class BartWithCustomEncoder(nn.Module):
    def __init__(self, pretrained_bart_model, custom_encoder):
        super(BartWithCustomEncoder, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_bart_model)
        self.bart.encoder = custom_encoder  # Replace the encoder
    
    def forward(self, graph_embeddings, decoder_input_ids, labels=None):
        # Pass graph embeddings through custom encoder
        encoder_outputs = self.bart.encoder(graph_embeddings)
        # Pass the outputs to BART decoder
        outputs = self.bart(
            inputs_embeds=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return outputs

# Step 3: Define the custom encoder parameters
graph_embedding_dim = 128  # Hypothetical input dimension for graph embeddings
hidden_dim = 512           # Intermediate hidden layer dimension
bart_hidden_dim = 768     # Must match BART's hidden dimension
custom_encoder = CustomGraphEncoder(graph_embedding_dim, hidden_dim, bart_hidden_dim)

# Step 4: Initialize the model
pretrained_bart_model = "facebook/bart-base"  # Can also use "facebook/bart-large"
model = BartWithCustomEncoder(pretrained_bart_model, custom_encoder)

# Step 5: Example usage
tokenizer = BartTokenizer.from_pretrained(pretrained_bart_model)

# Dummy graph embeddings (batch_size=2, sequence_length=3, input_dim=
dummy_graph_embeddings = torch.rand(2, 10, graph_embedding_dim)  # Batch size, sequence length, input_dim

# Dummy decoder inputs
decoder_input_ids = torch.tensor([[0, 50256, 50257], [0, 50258, 50259]])  # BOS and random tokens

# Forward pass
outputs = model(dummy_graph_embeddings, decoder_input_ids)

output_logits = outputs.logits

probabilities = torch.softmax(output_logits, dim=-1 )

token_ids = torch.argmax(probabilities, dim=-1)

print("Logits shape:", outputs.logits.shape)
print("Token ids shape:", token_ids.shape)
# Decode the output

decoded_outputs = tokenizer.batch_decode(token_ids, skip_special_tokens=True)

print("Decoded outputs:", decoded_outputs)
