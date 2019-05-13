# add package root
import os, sys
sys.path.insert(0, '../..')

from models.lstm_attn.model import LSTMAttnEncoderDecoder
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

    #train_loader.dataset.X = train_loader.dataset.X[0:300]
    #train_loader.dataset.y = train_loader.dataset.y[0:300]
    #valid_loader.dataset.X = valid_loader.dataset.X[0:300]
    #valid_loader.dataset.y = valid_loader.dataset.y[0:300]
    # ######################################################################
    
    # GPU SELECTION ########################################################
    if torch.cuda.is_available():
        freer_gpu = get_freer_gpu()
        print("Auto-selected GPU: " + str(freer_gpu))
        torch.cuda.set_device(freer_gpu)
    # ######################################################################
    
    # MODEL TRAINING #######################################################

    n_class = len(tgt_w2i)
    n_emb_dim = 300
    n_hidden = 128
    n_lstm_units = 2
    n_lstm_dropout = 0.2
    n_dropout = 0.3

    model = LSTMAttnEncoderDecoder(n_class, n_emb_dim, n_hidden, n_lstm_units, n_lstm_dropout, n_dropout)

    # ######################################################################
    max_epochs = 100    
    model_path = os.path.join("..", "..", "train", "lstm_attn")
    train(model, 
          src_i2w, 
          tgt_i2w,
          train_loader, 
          valid_loader, 
          test_loader,                          
          model_store_path=model_path, 
          resume=False, 
          max_epochs=max_epochs, 
          patience=10, 
          lr=0.0005)
    






































    # embedding_dim = 512
    # encoder_hidden_dim = 512
    # decoder_hidden_dim = 512
    # encoder_n_layers = 2
    # decoder_n_layers = 1
    # encoder_drop_prob = 0.3
    # decoder_drop_prob = 0.3
    # lr = 0.0005
    #
    # model = LSTMEncoderDecoderAtt(src_w2i, src_i2w, tgt_w2i, tgt_i2w, embedding_dim, encoder_hidden_dim,
    #                             decoder_hidden_dim, encoder_n_layers, decoder_n_layers,
    #                             encoder_drop_prob=encoder_drop_prob, decoder_drop_prob=decoder_drop_prob, lr=lr,
    #                             model_store_path="../../train/lstm_att")
    #
    # model.train(train_loader, valid_loader, test_loader, batch_size, patience=20)


# run
#net.load_checkpoint("best")
#input = [ [4,5,6,7,8,9], [9,8,7,6] ]
#output = net.run(input)
#output = net.run(valid_loader, batch_size)
#print(output)