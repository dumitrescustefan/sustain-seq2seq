import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(tokenizer.vocab_size)


text_1 = "Who was Jim Henson ? Jim Henson was a puppeteer"
text_2 = "Who was Jim Henson ? Jim Henson was a mysterious young man"
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_1)
indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text_2+[tokenizer.bos_token])
tokens_tensor_1 = torch.tensor([indexed_tokens1])
tokens_tensor_2 = torch.tensor([indexed_tokens2])

print(tokenized_text_1)
print(indexed_tokens1)
print(tokens_tensor_1)

print(tokenized_text_2)
print(indexed_tokens2)
print(tokens_tensor_2)

"""
print("Encode:")
text = "What is the fastest car in the "
indexed_tokens = tokenizer.encode(text)
# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])
print(indexed_tokens)
"""
encoder = GPT2Model.from_pretrained('gpt2')
with torch.no_grad():
    last_hidden_states_1, past = encoder(tokens_tensor_1)
print(last_hidden_states_1.size())


with torch.no_grad():
    last_hidden_states_2, past = encoder(tokens_tensor_2)
print(last_hidden_states_2.size())

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# Get the predicted next sub-word
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

# Print the predicted word
print(predicted_text)