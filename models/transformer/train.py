from data.roen.loader import loader
from models.transformer.model import Transformer
from models.util.trainer import train, get_freer_gpu
import os
import torch


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

    # train_loader.dataset.X = train_loader.dataset.X[0:100]
    # train_loader.dataset.y = train_loader.dataset.y[0:100]
    # valid_loader.dataset.X = valid_loader.dataset.X[0:100]
    # valid_loader.dataset.y = valid_loader.dataset.y[0:100]

    freer_gpu = get_freer_gpu()
    print("Selected GPU: " + str(freer_gpu))
    torch.cuda.set_device(freer_gpu)

    n_class = len(src_i2w)
    N = 6
    # embedding size
    d_model = 512
    # feed_forward size
    d_ff = 512
    # number of heads
    h = 8
    # embedding size for keys
    d_k = 64
    # embedding size for values
    d_v = 64
    transformer = Transformer(N, d_model, d_ff, h, d_k, d_v, n_class, max_seq_len)

    epochs = 100
    lr = 0.001
    model_path = os.path.join("..", "..", "train", "transformer.ckpt")
    train(transformer, epochs, batch_size, lr, n_class, train_loader, valid_loader, test_loader, src_i2w, tgt_i2w, model_path)
        
   