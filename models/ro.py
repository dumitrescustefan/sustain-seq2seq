import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import numpy as np
np.random.seed(0)
import random
random.seed(0)

import torch.nn.functional as F
import numpy as np
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import trange


from colorama import init
init(autoreset=True)
from colorama import Style, Fore

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
print(tokenizer.encode("Hello, my dog is cute"))

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
print(input_ids)
outputs = model(input_ids, labels=input_ids)
loss, logits = outputs[:2]
print(loss)
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
    
def sample_sequence(model, length, context, tokenizer, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    print("Sample")
    context = torch.tensor(context, dtype=torch.long, device=device)
    print(context)
    print(context.size())
    context = context.unsqueeze(0).repeat(num_samples, 1)    
    generated = context
    
    #generated = torch.tensor([[  464,  1664,   468,   257,  4939,  2450, 13593, 37388,   290,  9253, 13], [15496, 11, 616, 3290, 318, 13779]], dtype=torch.long, device=device)
    
    print(generated)
    print(generated.size())
    print("Generate::::::::::::::")
    with torch.no_grad():#labels=input_ids
        for i in trange(length):
            #print("Step "+str(i))
            input_ids = generated               
            #loss, output, past = model(input_ids, labels=labels)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            if i > 0:
                #loss, output, past = model(next_token.unsqueeze(0), labels=next_token.unsqueeze(0), past=past)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                output, past = model(next_token.unsqueeze(0), past=past)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                #output, past = model(generated)
                #print(generated.size())
                
            else:
                output, past = model(input_ids) #, labels=input_ids
            #print("----")
            #print(len(input_ids))
            #print("\noutput size: ")            
            
            #print()
            next_token_logits = output[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            print("Token:[{}]=[{}]".format(tokenizer.convert_ids_to_tokens(next_token.item()), next_token.item()))
            #print(generated.size())
            #print(next_token.unsqueeze(0).size())
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated
    
    
while True:
    raw_text = "A simple test"
    context_tokens = tokenizer.encode(raw_text)
    print(context_tokens)
    out = sample_sequence(
        model=model,
        context=context_tokens,
        tokenizer=tokenizer,
        length=50,
        temperature=1.,
        top_k=0,
        top_p=0.9,
        device=torch.device("cpu")
    )
    out = out[0, len(context_tokens):].tolist()
    text = tokenizer.decode(out, clean_up_tokenization_spaces=False)
    print("{}{}{}{}".format(Fore.GREEN,raw_text,Fore.RED,text))
    print("_"*20)
    break
