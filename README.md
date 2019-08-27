# sustain-seq2seq

- [ ] Tokenization that covers BPE and GPT2 (from pytorch_transformers) in a single Lookup object. Full tests for this are required as a lot of problems came from mismatched maps and out of range ints in the lookup.
  
  Encoding for BPE:
    X is composed of ints with <EOS> at the end, <PAD>s in rest
    y is <BOS> ints <EOS> <PAD>s 
  Encoding for GPT2:
    X and y are both <|endoftext|> ints <|endoftext|>s in rest (decoder should stop if <|endoftext|> is generated at index>1)
  
  
Models that need to work:
- [ ] LSTMEncoder + LSTMDecoder with Attention
- [ ] GPT2Encoder + LSTMDecoder with Attention
- [ ] LSTMEncoder + LSTMDecoder with Attention, Pointer Generator & Coverage 
- [ ] GPT2Encoder + LSTMDecoder with Attention, Pointer Generator & Coverage
- [ ] GPT2Encoder + GPT2Decoder with Pointer Generator & Coverage

Other stuff that needs to be done:
- [ ] Look at validation measures again (BLEU, METEOR, ROUGE)
- [ ] Implement all attention types (low priority)
- [ ] Experiment with multihead attention for RNNs
- [ ] Beamsearch and/or topk/topp as in pytorch_transformers
- [ ] Check attention masks are working everywhere
- [ ] Optimizer: Learning rate scheduler, superconvergence, warm restart si cyclical LR. Implement scheduler. Partially done, needs more testing.
