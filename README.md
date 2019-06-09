# sustain-seq2seq

## Road-map:

- [X] [Deprecated] Refactor models into components
- [X] Rewrite loaders with VariableLoader, make it compatible with Pytorch dataloader -> not worth it for now.
- [X] Rewrite training procedure with variable batch size for GPU training (avoid OOM errors) -> tried, not worth it for now.
- [ ] Pretrained word embeddings -> separate module maybe?

For LSTM Encoder-Decoder:

- [X] [Deprecated] Validate LSTM EncoderDecoder with AdditiveAttention
- [X] Add validation measures (BLEU, CIDEr, etc)
- [X] Add logging stuff + visual 
- [X] Add TeacherForcing & decay, validate it
- [ ] Add label smoothing
- [X] Validate global additive attention is working (plot heatmap for ro-en task)
- [ ] Implement attention with several types (class name: Attention) - in progress
- [ ] Encoder with self attention (multihead) (class name: ---)
- [ ] Decoder with self attention (multihead) (class name: ---)
- [ ] L2 weight regularization
- [ ] Add coverage (see NMT coverage), validate it
- [ ] Implement beam search with length penalty 
- [ ] Nice and clean run script, that takes strings as input and outputs strings back (i.e full process)
- [ ] Add temperature in the softmax of the decoder

For Transformer: 

- [ ] Is it actually working ? Take it from there.

Other:
- [X] add dummy reverse sequence dataset 
- [ ] Validation measures: input tokenization required? Validate it.
- [X] Attention: bias is needed for KQV transformations? Answer: guess not
- [X] Attention: bias is needed for computiation? Answer: Depends on case, mostly not
- [ ] Attention: mask needed before softmax? Partial answer: will speed up convergence, in the end won't really matter. Still on it.
- [ ] Transfer hidden=False not working, plus needs testing
- [ ] AdditiveAttention.py -> reimplement in Attention.py and delete the file
- [ ] Single loader in models/utils that loads lists of numbers + additional w2i file
- [ ] Tied embedding weights, investigate it
- [ ] Optimizer: Learning rate scheduler, superconvergence, warm restart si cyclical LR. Implement scheduler. *Priority*

## Validation scores:

- TODO

## Folder structure:

#### Data folder
In /data/ there are folders for each separate task. Each folder is self contained and has all the necessary files (including .py) that process that data into tensor files ready for input. It should contain .pt files (PyTorch tensors, split into train/dev/test) as well as source&target word2index/index2word dicts as jsons. 

Each folder should have a single loader.py that when called will offer a ``train, dev, test, src_w2i, src_i2w, tgt_w2i, tgt_i2w`` objects.
They should be called like: 
```
from data.<task>.loader import loader

data_folder = "../../data/<task>/<any_subfolder_with_pt_files>"
train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size, max_seq_len, min_seq_len)
```

#### Train folder 

This folder will contain training results. Each model is required to create subfolders in it.

#### Models folder

Each subfolder is a different model. Inputs are similar, imported with the ``loader`` function, output is similar to input - int tensors. 

Within each model the structure is:
```
/models/
  <model-eg. transformer>/
    train.py <- call it to train the model, parameters in the file
    run.py <- call it to run the model (inference only, with loaing of the model from the train folder)
    model.py <- this file will contain the main class of the model (e.g. class Transformer(), etc.)
    <any other file required>
```

To run one should call:
```
from models.<transformer>.model import Transformer
```
then do:
``` 
output = transformer_object(batched_dataloader_input)
```
