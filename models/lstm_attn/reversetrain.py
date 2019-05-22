# add package root
import os, sys
sys.path.insert(0, '../..')

from models.lstm_attn.model import LSTMEncoderDecoderWithAttention
from models.util.trainer import train, get_freer_gpu
import torch

if __name__ == "__main__":    
    
    # DATA PREPARATION ######################################################
    print("Loading data ...")
    from data.reverse.loader import loader    
    batch_size = 256
    min = 10
    max = 50
    vocab = 20
    train_count = 10000
    dev_count = 10
    train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader("", batch_size, max, min, train_count, dev_count, vocab)
    
    print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size src/tgt {}/{}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(test_loader.dataset.X),
        len(src_i2w), len(tgt_i2w)))
    X_sample, y_sample = iter(train_loader).next()
    
    print(X_sample[0])
    print(y_sample[0])
    # ######################################################################
    
    # GPU SELECTION ########################################################
    if torch.cuda.is_available():
        freer_gpu = get_freer_gpu()
        print("Auto-selected GPU: " + str(freer_gpu))
        torch.cuda.set_device(freer_gpu)
    # ######################################################################
    
    # MODEL TRAINING #######################################################
    
    model = LSTMEncoderDecoderWithAttention(
                enc_vocab_size=len(src_w2i),
                enc_emb_dim=30,
                enc_hidden_dim=100, # meaning we will have dim/2 for forward and dim/2 for backward lstm
                enc_num_layers=2,
                enc_dropout=0.2,
                enc_lstm_dropout=0.2,
                dec_input_dim=100, # must be equal to enc_hidden_dim
                dec_emb_dim=30,
                dec_hidden_dim=50,
                dec_num_layers=2,
                dec_dropout=0.2,
                dec_lstm_dropout=0.2,
                dec_vocab_size=len(tgt_w2i),
                dec_transfer_hidden=True)
    
    print("_"*80+"\n")
    print(model)
    print("_"*80+"\n")
    
    train(model, 
          src_i2w, 
          tgt_i2w,
          train_loader, 
          valid_loader,
          test_loader,                          
          model_store_path = os.path.join("..", "..", "train", "lstm_attn"), 
          resume = False, 
          max_epochs = 30, 
          patience = 30, 
          lr = 0.001,
          tf_start_ratio=1.,
          tf_end_ratio=0.1,
          tf_epochs_decay=10)
          
    # ######################################################################