from data.roen.loader import loader
from models.transformer.model import Transformer
from models.util.trainer import train
import os


if __name__ == "__main__":
    data_folder = os.path.join("..", "..", "data", "roen", "setimes.8K.bpe")

    batch_size = 32
    min_seq_len = 10
    max_seq_len = 10000

    print("Loading data ...")
    train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size, max_seq_len, min_seq_len)
    
    print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size {}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(test_loader.dataset.X),
        len(tgt_i2w)))

    # train_loader.dataset.X = train_loader.dataset.X[0:1000]
    # train_loader.dataset.y = train_loader.dataset.y[0:1000]
    # valid_loader.dataset.X = valid_loader.dataset.X[0:1000]
    # valid_loader.dataset.y = valid_loader.dataset.y[0:1000]

    n_class = len(src_i2w)
    # number of enc/dec
    N = 2
    # embedding size
    d_model = 128
    # feed_forward size
    d_ff = 512
    # number of heads
    h = 4
    # embedding size for keys
    d_k = 32
    # embedding size for values
    d_v = 32
    transformer = Transformer(N, d_model, d_ff, h, d_k, d_v, n_class, max_seq_len)

    epochs = 10
    lr = 0.01
    train(transformer, epochs, batch_size, lr, n_class, train_loader, valid_loader, test_loader, src_i2w, tgt_i2w)
        
   