from models.transformer.model import Transformer
from models.util.trainer import train, get_freer_gpu
import os
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

    train_loader.dataset.X = train_loader.dataset.X[0:100]
    train_loader.dataset.y = train_loader.dataset.y[0:100]
    valid_loader.dataset.X = valid_loader.dataset.X[0:100]
    valid_loader.dataset.y = valid_loader.dataset.y[0:100]
    # ######################################################################
    
    # GPU SELECTION ########################################################
    if torch.cuda.is_available():
        freer_gpu = get_freer_gpu()
        print("Auto-selected GPU: " + str(freer_gpu))
        torch.cuda.set_device(freer_gpu)
    # ######################################################################
    
    # MODEL PARAMETERS #####################################################
    n_class = len(tgt_i2w)
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
    model = Transformer(N, d_model, d_ff, h, d_k, d_v, n_class, max_seq_len)

    # ######################################################################
    max_epochs = 100    
    model_path = os.path.join("..", "..", "train", "transformer")
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
    