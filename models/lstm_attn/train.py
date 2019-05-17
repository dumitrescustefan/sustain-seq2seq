# add package root
import os, sys
sys.path.insert(0, '../..')

from models.lstm_attn.model import LSTMEncoderDecoderWithAdditiveAttention
from models.util.trainer import train, get_freer_gpu
import torch

if __name__ == "__main__":    
    
    # DATA PREPARATION ######################################################
    print("Loading data ...")
    batch_size = 32
    min_seq_len = 10
    max_seq_len = 10000

    from data.roen.loader import loader
    data_folder = os.path.join("..", "..", "data", "roen", "ready", "setimes.8K.bpe")
    train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size, max_seq_len, min_seq_len)
    
    print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size src/tgt {}/{}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(test_loader.dataset.X),
        len(src_i2w), len(tgt_i2w)))

    # train_loader.dataset.X = train_loader.dataset.X[0:300]
    # train_loader.dataset.y = train_loader.dataset.y[0:300]
    # valid_loader.dataset.X = valid_loader.dataset.X[0:100]
    # valid_loader.dataset.y = valid_loader.dataset.y[0:100]
    # ######################################################################
    
    # GPU SELECTION ########################################################
    if torch.cuda.is_available():
        freer_gpu = get_freer_gpu()
        print("Auto-selected GPU: " + str(freer_gpu))
        torch.cuda.set_device(freer_gpu)
    # ######################################################################
    
    # MODEL TRAINING #######################################################
    
    model = LSTMEncoderDecoderWithAdditiveAttention(
                enc_vocab_size=len(src_w2i),
                enc_emb_dim=300,
                enc_hidden_dim=512, # meaning we will have dim/2 for forward and dim/2 for backward lstm
                enc_num_layers=2,
                enc_dropout=0.2,
                enc_lstm_dropout=0.2,
                dec_input_dim=56, # must be equal to enc_hidden_dim
                dec_emb_dim=300,
                dec_hidden_dim=256,
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
          resume = True, 
          max_epochs = 100, 
          patience = 10, 
          lr = 0.001,
          tf_start_ratio=1.,
          tf_end_ratio=0.1,
          tf_epochs_decay=10)
          
    # ######################################################################