# sustain-seq2seq

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
