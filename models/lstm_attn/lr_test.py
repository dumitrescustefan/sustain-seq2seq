# add package root
import os, sys
sys.path.insert(0, '../..')

from models.lstm_attn.model import LSTMEncoderDecoderWithAttention
from models.util.trainer import train, get_freer_gpu
from models.util.lr_range_test import LRFinder
import torch
import torch.nn as nn

if __name__ == "__main__": 

# DATA PREPARATION ######################################################
    print("Loading data ...")
     # CMUDICT test
    batch_size = 64
    min_seq_len_X = 0
    max_seq_len_X = 10000
    min_seq_len_y = min_seq_len_X
    max_seq_len_y = max_seq_len_X
    from models.util.loaders.standard import loader
    data_folder = os.path.join("..", "..", "data", "cmudict", "ready")
    src_w2i = "X_word2index.json"
    src_i2w = "X_index2word.json"
    tgt_w2i = "y_word2index.json"
    tgt_i2w = "y_index2word.json"
    train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size, src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y)
    
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
    
    model = LSTMEncoderDecoderWithAttention(
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
                dec_attention_type = "additive",
                dec_transfer_hidden=True)
    
    print("_"*80+"\n")
    print(model)
    print("_"*80+"\n")


    criterion = nn.CrossEntropyLoss()#ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #optim.Adam(net.parameters(), lr=1e-7, weight_decay=1e-2)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-3)
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"        
    
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=400)
    lr_finder.plot()#, log_lr=False)
