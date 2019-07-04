# add package root
import os, sys
sys.path.insert(0, '../..')

from models.nlg_simple_lstm.model import NLG_SimpleEncoder_LSTMDecoderWithAttentionAndSelfAttention
from models.util.trainer import train, get_freer_gpu
from data.e2e.data import Slot, Slots

import torch

if __name__ == "__main__":    
    
    # DATA PREPARATION ######################################################
    print("Loading data ...")
    batch_size = 32
    
    from data.e2e.loader_vector import loader
    data_folder = os.path.join("..", "..", "data", "e2e", "vector")
    slots, train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size)
    
    slot_sizes = []
    for slot in slots.slots:
        slot_sizes.append(len(slot.values))
    
    print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size tgt {}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(test_loader.dataset.X),
        len(tgt_i2w)))

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
    model = NLG_SimpleEncoder_LSTMDecoderWithAttentionAndSelfAttention(                
                enc_emb_dim=32,
                slot_sizes = slot_sizes,
                enc_dropout=0.2,
                dec_input_dim=32,  
                dec_emb_dim=256,
                dec_hidden_dim=256,
                dec_num_layers=1,
                dec_dropout=0.33,
                dec_lstm_dropout=0.33,
                dec_vocab_size=len(tgt_w2i),
                dec_attention_type = "coverage")
    
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
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)#, weight_decay=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum = 0.9)
    lr_scheduler = None
    
    train(model, 
          src_i2w, 
          tgt_i2w,
          train_loader, 
          valid_loader,
          test_loader,                          
          model_store_path = os.path.join("..", "..", "train", "nlg_simple_lstm"), 
          resume = False, 
          max_epochs = 300, 
          patience = 25, 
          optimizer = optimizer,
          lr_scheduler = lr_scheduler,
          tf_start_ratio=0.9,
          tf_end_ratio=0.1,
          tf_epochs_decay=50)
          
    # ######################################################################