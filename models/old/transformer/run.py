from models.util.runner import run
import os
from data.roen.loader import loader
import torch


if __name__ == "__main__":
    data_folder = os.path.join("..", "..", "data", "roen", "setimes.8K.bpe")

    batch_size = 1
    min_seq_len = 10
    max_seq_len = 10000

    print("Loading data ...")
    train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size,
                                                                                         max_seq_len, min_seq_len)

    print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size {}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(test_loader.dataset.X),
        len(tgt_i2w)))

    test_loader.dataset.X = train_loader.dataset.X[0:100]
    test_loader.dataset.y = train_loader.dataset.y[0:100]

    print("Loading the model...")
    model_path = os.path.join("..", "..", "train", "transformer.ckpt")
    model = torch.load(model_path)
    print("Loading model successfully\n")

    run(model, test_loader, src_i2w, tgt_i2w)