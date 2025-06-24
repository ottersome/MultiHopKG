import pickle
from transformers.models.bart import BartTokenizer
import random
import torch

# Load the cache in .cache/valid.pkl
cache_path = ".cache/valid.pkl"
num_cache_samples = 10
seed = 42   

# Set the seed for random sampling
torch.manual_seed(seed)

with open(cache_path, "rb") as f:
    cache = pickle.load(f)
    print(f"Loaded the cache with {len(cache)} samples. With columsn {cache.columns}")

print(f"THis is what the cache looks like: {cache}")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
sep_token = tokenizer.sep_token_id
padding_value = tokenizer.pad_token_id

merged_questions_answers = []
for question, answer in zip(cache["enc_questions"], cache["enc_answer"]):
    qna = question + [sep_token] + answer
    merged_questions_answers.append(torch.tensor(qna))

# Now we pad them
qna_padded = torch.nn.utils.rnn.pad_sequence(
    merged_questions_answers, batch_first=True, padding_value=padding_value
)

# Tokenize a string
tokens_test = tokenizer("Hello, my dog is cute")
print(tokens_test)
tokens_test = torch.tensor(tokens_test.input_ids).reshape(1, -1).repeat(2, 1)
print(tokens_test)
# Lets decode them 
decoded = tokenizer.batch_decode(tokens_test)
print(decoded)

# Get `num_cache_samples` random samples from the cache
random_samples = torch.randint(0, len(qna_padded), (num_cache_samples,))
random_samples = torch.index_select(qna_padded, 0, random_samples)

print(f"Random samples: {random_samples}")

for sample in random_samples:
    print("----------------------------------------")
    print(tokenizer.decode(sample, skip_special_tokens=True))
    print("----------------------------------------")
