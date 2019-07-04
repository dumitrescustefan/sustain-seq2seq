# add package root
import os, sys
sys.path.insert(0, '../..')

from models.lstm_selfattn.model import LSTMEncoderDecoderWithAttentionAndSelfAttention
from models.util.trainer import train, get_freer_gpu
import torch

if __name__ == "__main__":    
    
    # DATA PREPARATION ######################################################
    print("Loading data ...")
    batch_size = 256
    min_seq_len_X = 10
    max_seq_len_X = 50
    min_seq_len_y = min_seq_len_X
    max_seq_len_y = max_seq_len_X

    #from data.roen.loader import loader
    #data_folder = os.path.join("..", "..", "data", "roen", "ready", "setimes.8K.bpe")
    #from data.fren.loader import loader
    from models.util.loaders.standard import loader
    data_folder = os.path.join("..", "..", "data", "fren", "ready")
    train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y)
    
    print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size src/tgt {}/{}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(test_loader.dataset.X),
        len(src_i2w), len(tgt_i2w)))

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
    
    model = LSTMEncoderDecoderWithAttentionAndSelfAttention(
                enc_vocab_size=len(src_w2i),
                enc_emb_dim=256,
                enc_hidden_dim=512, # meaning we will have dim/2 for forward and dim/2 for backward lstm
                enc_num_layers=1,
                enc_dropout=0.33,
                enc_lstm_dropout=0.33,
                dec_input_dim=256, 
                dec_emb_dim=256,
                dec_hidden_dim=256,
                dec_num_layers=1,
                dec_dropout=0.33,
                dec_lstm_dropout=0.33,
                dec_vocab_size=len(tgt_w2i),
                dec_attention_type = "coverage",
                dec_transfer_hidden=True)
    
    print("_"*80+"\n")
    print(model)
    print("_"*80+"\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)#, weight_decay=1e-3)
    lr_scheduler = None
    
    train(model, 
          src_i2w, 
          tgt_i2w,
          train_loader, 
          valid_loader,
          test_loader,                          
          model_store_path = os.path.join("..", "..", "train", "lstm_selfattn"), 
          resume = False, 
          max_epochs = 500, 
          patience = 25, 
          optimizer = optimizer,
          lr_scheduler = lr_scheduler,
          tf_start_ratio=0.9,
          tf_end_ratio=0.1,
          tf_epochs_decay=50)
     
          
    # ######################################################################