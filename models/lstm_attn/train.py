# add package root
import os, sys
sys.path.insert(0, '../..')


from data.roen.loader import loader

from models.lstm_attn.model import LSTMAttnEncoderDecoder
from models.util.trainer import train

if __name__ == "__main__":
    data_folder = os.path.join("..", "..", "data", "roen", "setimes.8K.bpe")

    batch_size = 15
    min_seq_len = 5
    max_seq_len = 10000

    print("Loading data ...")
    train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size, max_seq_len, min_seq_len)
    
    print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size {}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(test_loader.dataset.X),
        len(src_w2i)))

    train_loader.dataset.X = train_loader.dataset.X[0:300]
    train_loader.dataset.y = train_loader.dataset.y[0:300]
    valid_loader.dataset.X = valid_loader.dataset.X[0:300]
    valid_loader.dataset.y = valid_loader.dataset.y[0:300]

    n_class = len(src_w2i)
    n_emb_dim = 300
    n_hidden = 128
    n_lstm_units = 2
    n_lstm_dropout = 0.2
    n_dropout = 0.3

    model = LSTMAttnEncoderDecoder(n_class, n_emb_dim, n_hidden, n_lstm_units, n_lstm_dropout, n_dropout)

    epochs = 500
    lr = 0.01
    train(model, epochs, batch_size, lr, n_class, train_loader, valid_loader, test_loader, src_i2w, tgt_i2w)







































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