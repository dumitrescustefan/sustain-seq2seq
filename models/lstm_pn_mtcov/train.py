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

from models.util.trainer import train
from models.util.lookup import Lookup
from models.util.loaders.standard import loader
from models.util.utils import select_processing_device

from models.lstm_pn.model import PNEncoderDecoder
from models.components.encoders.LSTMEncoder import Encoder
from models.components.decoders.LSTMDecoder_Att_PN_MTCov import Decoder

if __name__ == "__main__":    
   
    # DATA PREPARATION ######################################################
    print("Loading data ...")
    """ FR-EN 
    batch_size = 8
    min_seq_len_X = 0
    max_seq_len_X = 500
    min_seq_len_y = min_seq_len_X
    max_seq_len_y = max_seq_len_X    
    data_folder = os.path.join("..", "..", "data", "fren", "ready", "bpe")
    src_lookup_prefix = os.path.join("..", "..", "data", "fren", "lookup", "bpe","src-4096")
    tgt_lookup_prefix = os.path.join("..", "..", "data", "fren", "lookup", "bpe","tgt-4096")    
    """
    """ CMU DICT """
    batch_size = 4
    min_seq_len_X = 0
    max_seq_len_X = 1000
    min_seq_len_y = min_seq_len_X
    max_seq_len_y = max_seq_len_X    
    #data_folder = os.path.join("..", "..", "data", "cmudict", "ready", "bpe")
    #src_lookup_prefix = os.path.join("..", "..", "data", "cmudict", "lookup", "bpe","src-256")
    #tgt_lookup_prefix = os.path.join("..", "..", "data", "cmudict", "lookup", "bpe","tgt-256")
    data_folder = os.path.join("..", "..", "data", "task2", "ready", "gpt2")
    src_lookup_prefix = os.path.join("..", "..", "data", "task2", "lookup", "gpt2","src")
    tgt_lookup_prefix = os.path.join("..", "..", "data", "task2", "lookup", "gpt2","tgt")
    
    
    src_lookup = Lookup(type="gpt2")
    src_lookup.load(src_lookup_prefix)
    tgt_lookup = Lookup(type="gpt2")
    tgt_lookup.load(tgt_lookup_prefix)
    train_loader, valid_loader, test_loader = loader(data_folder, batch_size, src_lookup, tgt_lookup, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y, custom_filename_prefix = "Business_Ethics_")
    
    #xXXXXXXXXXX
    #valid_loader = train_loader
    
    print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size src/tgt {}/{}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(test_loader.dataset.X),
        len(src_lookup), len(tgt_lookup)))
    # ######################################################################
    
    # GPU SELECTION ########################################################
    device = select_processing_device(verbose = True)
    # ######################################################################
    
    # MODEL TRAINING #######################################################
    
    aux_loss_weight = 0.03
    
    encoder = Encoder(
                vocab_size=len(src_lookup),
                emb_dim=300,
                hidden_dim=512, # meaning we will have dim/2 for forward and dim/2 for backward lstm
                num_layers=2,
                lstm_dropout=0.4,
                dropout=0.4,
                device=device)
    decoder = Decoder(                
                emb_dim=300,
                input_size=512,                 
                hidden_dim=256,
                num_layers=2,
                lstm_dropout=0.4,
                dropout=0.4,
                vocab_size=len(tgt_lookup),                
                device=device)
        
    model = PNEncoderDecoder(src_lookup = src_lookup, tgt_lookup = tgt_lookup, encoder = encoder, decoder = decoder, dec_transfer_hidden = True, aux_loss_weight = aux_loss_weight, device = device)
                
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
          model_store_path = os.path.join("..", "..", "train", "lstm_pn"), 
          resume = False, 
          max_epochs = 400, 
          patience = 25, 
          optimizer = optimizer,
          lr_scheduler = lr_scheduler,
          tf_start_ratio=1.0,
          tf_end_ratio=0.1,
          tf_epochs_decay=50)
          
    # ######################################################################