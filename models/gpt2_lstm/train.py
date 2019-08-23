# add package root
import os, sys
sys.path.insert(0, '../..')

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import numpy as np
np.random.seed(0)
import random
random.seed(0)


from models.gpt2_lstm.model import CustomEncoderDecoder
from models.util.trainer import train, get_freer_gpu
from models.util.lookup import Lookup
from models.util.loaders.standard import loader
import torch

if __name__ == "__main__":    
   
    # DATA PREPARATION ######################################################
    print("Loading data ...")
    batch_size = 8
    min_seq_len_X = 0
    max_seq_len_X = 500
    min_seq_len_y = min_seq_len_X
    max_seq_len_y = max_seq_len_X    
    data_folder = os.path.join("..", "..", "data", "fren", "ready", "bpe")
    src_lookup_prefix = os.path.join("..", "..", "data", "fren", "lookup", "bpe","src-4096")
    tgt_lookup_prefix = os.path.join("..", "..", "data", "fren", "lookup", "bpe","tgt-4096")
    src_lookup = Lookup(type="bpe")
    src_lookup.load(src_lookup_prefix)
    tgt_lookup = Lookup(type="bpe")
    tgt_lookup.load(tgt_lookup_prefix)
    train_loader, valid_loader, test_loader = loader(data_folder, batch_size, src_lookup, tgt_lookup, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y)
    
    
    # FR-EN test
    """
    batch_size = 4
    min_seq_len_X = 5
    max_seq_len_X = 15
    min_seq_len_y = min_seq_len_X
    max_seq_len_y = max_seq_len_X
    from models.util.loaders.standard import loader
    data_folder = os.path.join("..", "..", "data", "fren", "ready")
    src_w2i = "fr_word2index.json"
    src_i2w = "fr_index2word.json"
    tgt_w2i = "en_word2index.json"
    tgt_i2w = "en_index2word.json"
    train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size, src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y)
    
   
    
    # CMUDICT test
    batch_size = 16#5
    min_seq_len_X = 0
    max_seq_len_X = 7 
    min_seq_len_y = min_seq_len_X
    max_seq_len_y = max_seq_len_X
    from models.util.loaders.standard import loader
    data_folder = os.path.join("..", "..", "data", "cmudict", "ready")
    src_w2i = "X_word2index.json"
    src_i2w = "X_index2word.json"
    tgt_w2i = "y_word2index.json"
    tgt_i2w = "y_index2word.json"
    train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size, src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y)
    """
    
    print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size src/tgt {}/{}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(test_loader.dataset.X),
        len(src_lookup), len(tgt_lookup)))

    #train_loader.dataset.X = train_loader.dataset.X[0:800]
    #train_loader.dataset.y = train_loader.dataset.y[0:800]
    #valid_loader.dataset.X = valid_loader.dataset.X[0:100]
    #valid_loader.dataset.y = valid_loader.dataset.y[0:100]
    # ######################################################################
    
    # GPU SELECTION ########################################################
    if torch.cuda.is_available():
        freer_gpu = get_freer_gpu()
        print("Auto-selected GPU: " + str(freer_gpu))
        torch.cuda.set_device(freer_gpu)
    # ######################################################################
    
    # MODEL TRAINING #######################################################
        
    model = CustomEncoderDecoder(
                src_lookup = src_lookup,
                tgt_lookup = tgt_lookup,
                enc_vocab_size=len(src_lookup),
                enc_emb_dim=300,
                enc_hidden_dim=1024, # meaning we will have dim/2 for forward and dim/2 for backward lstm
                enc_num_layers=2,
                enc_dropout=0.4,
                enc_lstm_dropout=0.4,
                dec_input_dim=256, # same as enc_hidden_dim
                dec_emb_dim=300,
                dec_hidden_dim=256,
                dec_num_layers=2,
                dec_dropout=0.4,
                dec_lstm_dropout=0.4,
                dec_vocab_size=len(tgt_lookup),
                dec_attention_type = "additive",
                dec_transfer_hidden = False)
                
    print("_"*80+"\n")
    print(model)
    print("_"*80+"\n")
    
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=1., momentum=0.9)
    from models.util.lr_scheduler import cyclical_lr
    end_lr = 500.
    step_size = len(train_loader)
    factor = 4
    clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr) #, decay_factor_per_step=.97)
    print("Step-size: {}, lr: {} -> {}".format(step_size, end_lr/factor, end_lr))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)#, weight_decay=1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=0.9)
    
    lr_scheduler = None
    
    #from models.components.criteria.SmoothedCrossEntropyLoss import SmoothedCrossEntropyLoss
    #criterion = SmoothedCrossEntropyLoss(ignore_index=0, label_smoothing=0.9)    
    #criterion = SmoothedCrossEntropyLoss(label_smoothing=1.) # simple crossentropy, no ignore index set
    #criterion = SmoothedCrossEntropyLoss(ignore_index=0, label_smoothing=1.) # simple crossentropy, with ignore index set
    criterion = nn.NLLLoss(ignore_index=tgt_lookup.convert_tokens_to_ids(tgt_lookup.pad_token))
    
    
    train(model, 
          train_loader, 
          valid_loader,
          test_loader,                          
          model_store_path = os.path.join("..", "..", "train", "lstm_attn_pn"), 
          resume = False, 
          max_epochs = 400, 
          patience = 25, 
          optimizer = optimizer,
          lr_scheduler = lr_scheduler,
          tf_start_ratio=1.0,
          tf_end_ratio=0.1,
          tf_epochs_decay=50)
          
    # ######################################################################