import sys
sys.path.insert(0, '../../..')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers import GPT2LMHeadModel
from models.components.attention.SummaryCoverageAttention import Attention

class Decoder(nn.Module):
    def __init__(self, lookup, input_size, top_k, top_p, device):
        """ 
            Creates a Decoder with attention and Pointer network see https://nlp.stanford.edu/pubs/see2017get.pdf 
        """        
        super().__init__()
        
        self.device = device
        
        self.gpt2lmheadmodel = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2lmheadmodel.resize_token_embeddings(len(lookup))
        for param in self.gpt2lmheadmodel.parameters():
            param.requires_grad = False
            
        self.lookup = lookup
        self.emb_dim = 768
        self.hidden_dim = 768
        self.vocab_size = len(lookup)
        self.encoder_size = input_size
        self.top_k = top_k
        self.top_p = top_p
                
        self.output_linear = nn.Linear(hidden_dim, vocab_size)
        self.attention = Attention(encoder_size=input_size, decoder_size=self.hidden_dim, vocab_size=vocab_size, device=device)

        # overwrite output to allow context from the attention to be added to the output layer
        self.output_linear = nn.Linear(self.hidden_dim+self.encoder_size+self.emb_dim, int((self.hidden_dim+self.encoder_size+self.emb_dim)/2))
        self.vocab_linear = nn.Linear(int((self.hidden_dim+self.encoder_size+self.emb_dim)/2), self.vocab_size)

        self.p_gen_linear = nn.Linear(self.encoder_size + self.hidden_dim*2 + self.emb_dim, 1)
        
        self.to(device)

    def forward(self, x_tuple, y_tuple, enc_output, teacher_forcing_ratio):
        
        src, src_lengths, src_masks = x_tuple[0], x_tuple[1], x_tuple[2]
        tgt, tgt_lengths, tgt_masks = y_tuple[0], y_tuple[1], y_tuple[2]
        
        batch_size = tgt.shape[0]
        src_seq_len = src.shape[1]
        seq_len_dec = tgt.shape[1]        
        attention_weights = []
        
        dec_states = (dec_states[0].contiguous(), dec_states[1].contiguous())
        output = torch.zeros(batch_size,seq_len_dec-1,self.vocab_size).to(self.device)
        
        coverage = torch.zeros(batch_size, self.vocab_size).to(self.device)
        coverage_loss = 0
        
        
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        
        # Loop over the rest of tokens in the tgt seq_len_dec.
        for i in range(0, seq_len_dec-1):
            # Calculate the context vector at step i.
            # context_vector is [batch_size, encoder_size], attention_weights is [batch_size, src_seq_len, 1], coverage is [batch_size, vocab_size]
            context_vector, step_attention_weights  = self.attention(state_h=dec_states[0], enc_output=enc_output, coverage=coverage, mask=src_masks)
            
            # save attention weights incrementally
            attention_weights.append(step_attention_weights.squeeze(2).cpu().tolist())
            
            if np.random.uniform(0, 1) < teacher_forcing_ratio or i is 0: # forces correct input word
                prev_output_embeddings = self.dropout(self.embedding(tgt[:, i]))               
            else: # takes the last generated word
                prev_output_embeddings = self.dropout(self.embedding(torch.squeeze(torch.argmax(vocab_logits, dim=2), dim=1)))
                
            lstm_input = torch.cat((prev_output_embeddings, context_vector), dim=1).reshape(batch_size, 1, -1)

            # generate next word for each instance in batch
            with torch.no_grad():
                inputs = {'input_ids': generated}
           
                outputs = model(**inputs)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)


            lin_input = torch.cat( (dec_output, context_vector.unsqueeze(1), prev_output_embeddings.unsqueeze(1)) , dim = 2)
            lin_output = self.output_linear(lin_input) 
            
            # vocab_dist is the softmaxed dist of the output of the generator, and is [batch_size, vocab_size]
            vocab_logits = self.vocab_linear(torch.tanh(lin_output))
            vocab_dist = torch.softmax(vocab_logits.squeeze(1), dim=1)
            
            # Calculate p_gen -> [batch_size, 1]            
            # context_vector is [batch_size, encoder_size]
            # dec_states[-1][1] is [batch_size, decoder_size]
            # prev_output_embeddings is [batch_size, emb_dim]            
            p_gen_input = torch.cat( (context_vector, dec_states[-1][0], dec_states[-1][1], prev_output_embeddings) , dim = 1)
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input)) 
            
            # Calculate final distribution, final_dist will be [batch_size, vocab_size]
            # vocab_dist is [batch_size, vocab_size], step_attention_weights is [batch_size, src_seq_len, 1], src is [batch_size, src_seq_len] and contains indices
            # first, we must use step_attention_weights to get attention_dist to be [batch_size, vocab_size]
            attention_dist = torch.zeros(batch_size, self.vocab_size).to(self.device)
            attention_dist = attention_dist.scatter_add(1, src, step_attention_weights.squeeze(2))
            
            #print("Step {}, \tp_gen is {:.4f}\t, y is {}, generated: {}".format(i, p_gen[0].item(), tgt[0, i].item(), torch.argmax(vocab_logits, dim=2)[0].item()))
            final_dist = p_gen * vocab_dist + (1-p_gen) * attention_dist
            
            # Adds the current output to the final output. 
            #output = torch.cat((output, lin_output), dim=1)            
            output[:,i,:] = final_dist #softmax_output.squeeze(1)
            
                       
            # update coverage loss, both are [batch_size, vocab_size]
            coverage_loss = coverage_loss + torch.sum(torch.min(attention_dist, coverage))/batch_size
                      
            
            #print("Step {}, coverage:  {}, cov_loss {}".format(i, torch.sum(coverage), coverage_loss))
            #print(torch.sum(coverage-attention_dist))
            #for q in range(self.vocab_size):
            #    print("{} - {}\t min={}".format(coverage[0][q], attention_dist[0][q], torch.min(coverage[0][q], attention_dist[0][q])))
            
            # calculate the next coverage by adding step_attention_weights where appropriate                        
            coverage = coverage.scatter_add(1, src, step_attention_weights.squeeze(2))
            
        # output is a tensor [batch_size, seq_len_dec, vocab_size], log-ged to be prepared for NLLLoss 
        # attention_weights is a list of [batch_size, seq_len] elements, where each element is the softmax distribution for a timestep
        # coverage_loss is a scalar tensor
        return {'output':torch.log(output + 1e-31), 'attention_weights':attention_weights, 'coverage_loss':coverage_loss}
    
    
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
    
def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
           
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated
    
    
while True:
    raw_text = "The company has a weak policy addressing bribery and corruption."
    context_tokens = tokenizer.encode(raw_text)
    out = sample_sequence(
        model=model,
        context=context_tokens,
        length=200,
        temperature=1.,
        top_k=0,
        top_p=0.9,
        device=torch.device("cpu")
    )
    out = out[0, len(context_tokens):].tolist()
    text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
    print(text)
    print("_"*20)
