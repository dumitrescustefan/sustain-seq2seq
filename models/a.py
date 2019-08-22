import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(tokenizer.vocab_size)


#special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'unk_token': '<UNK>', 'sep_token': '<SEP>', 'pad_token': '<PAD>', 'cls_token' : '<CLS>'}
#num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
#print('We have added', num_added_toks, 'tokens')
#model.resize_token_embeddings(len(tokenizer))
print(tokenizer.special_tokens_map)

# Encode a text inputs


print(tokenizer.convert_tokens_to_ids("What is"))
print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([2061, 318])))
print(tokenizer.all_special_ids)
print(tokenizer.unk_token)

print("Encode:")
text = "What is the fastest car in the "
indexed_tokens = tokenizer.encode(text)
# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])
print(indexed_tokens)

encoder = GPT2Model.from_pretrained('gpt2')
with torch.no_grad():
    last_hidden_states = encoder(tokens_tensor)[0]
print(last_hidden_states.size())



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