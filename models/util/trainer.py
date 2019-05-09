import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os


def get_freer_gpu():
    import os
    import numpy as np
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))


def _print_some_examples(model, loader, seq_len, src_i2w, tgt_i2w):
    X_sample, y_sample = iter(loader).next()
    X_sample = X_sample[0:seq_len]
    y_sample = y_sample[0:seq_len]
    if model.cuda:
        X_sample = X_sample.cuda()
        y_sample = y_sample.cuda()

    y_pred_dev_sample = model.forward(X_sample, y_sample)
    y_pred_dev_sample = torch.argmax(y_pred_dev_sample, dim=2)

    for i in range(seq_len):
        for j in range(len(X_sample[i])):
            token = str(X_sample[i][j].item())

            if token not in src_i2w.keys():
                print(src_i2w['1'] + " ", end='')
            elif token == '3':
                print(src_i2w['3'], end='')
                break
            else:
                print(src_i2w[token] + " ", end='')
        print()
        for j in range(len(y_sample[i])):
            token = str(y_sample[i][j].item())

            if token not in tgt_i2w.keys():
                print(tgt_i2w['1'] + " ", end='')
            elif token == '3':
                print(tgt_i2w['3'], end='')
                break
            else:
                print(tgt_i2w[token] + " ", end='')
        print()
        for j in range(len(y_pred_dev_sample[i])):
            token = str(y_pred_dev_sample[i][j].item())

            if token not in tgt_i2w.keys():
                print(tgt_i2w['1'] + " ", end='')
            elif token == '3':
                print(tgt_i2w['3'], end='')
                break
            else:
                print(tgt_i2w[token] + " ", end='')
        print()
        print("-" * 40)


def train(model, epochs, batch_size, lr, n_class, train_loader, valid_loader, test_loader, src_i2w, tgt_i2w, model_path):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_data = len(train_loader.dataset.X)
    n_dev_data = len(valid_loader.dataset.X)

    # training
    for epoch in range(epochs):
        model.train()

        average_loss = 0
        cnt = 0

        t = tqdm(train_loader, desc="Epoch " + str(epoch), unit="batches")
        for (x_batch, y_batch) in t:
            if model.cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            y_in_batch = y_batch[:, :-1]
            y_out_batch = y_batch[:, 1:]

            # if batch_counter%100==0:
            #    print("Batch: {}/{}".format(batch_counter, int(n_data/batch_size)))

            optimizer.zero_grad()

            output = model.forward(x_batch, y_in_batch)

            loss = criterion(output.view(-1, n_class), y_out_batch.contiguous().view(-1))
            average_loss += loss

            # if batch_counter%100==0:
            #    print("Loss: {}".format(loss))
            loss.backward()
            optimizer.step()

            cnt += 1
            t.set_postfix(loss=average_loss.data.item() / cnt)  # print loss
            # _print_some_examples(model, test_loader, 1)

        # deving
        model.eval()
        _print_some_examples(model, test_loader, batch_size, src_i2w, tgt_i2w)
        dev_accuracy = 0

        for x_dev_batch, y_dev_batch in valid_loader:
            if model.cuda:
                x_dev_batch = x_dev_batch.cuda()
                y_dev_batch = y_dev_batch.cuda()

            y_in_dev_batch = y_dev_batch[:, :-1]
            y_out_dev_batch = y_dev_batch[:, 1:]

            y_pred_dev = model.forward(x_dev_batch, y_in_dev_batch).argmax(dim=2).view(-1)

            dev_accuracy += accuracy_score(y_out_dev_batch.contiguous().view(-1).cpu(), y_pred_dev.cpu())

        print("\nValidation Accuracy: {}".format(dev_accuracy / (n_dev_data / batch_size)))
        print("Average loss: {}\n".format(average_loss / (n_data / batch_size)))

    print("Saving the model...")
    torch.save(model, model_path)
    print("Model saved successfully")
